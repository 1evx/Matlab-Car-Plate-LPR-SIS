function [rectifiedPlate, metadata] = rectifyPlate(inputImage, bbox, config, hints)
    %RECTIFYPLATE Crop and deskew a detected plate region.

    if nargin < 4 || isempty(hints)
        hints = struct();
    end

    config = validateConfig(config);
    paddedBBox = localPadBBox(bbox, size(inputImage), config.rectification.cropPaddingRatio);
    cropped = imcrop(inputImage, paddedBBox);
    if isempty(cropped)
        rectifiedPlate = [];
        metadata = localEmptyMetadata();
        return;
    end

    gray = im2gray(cropped);
    enhanced = enhanceContrast(reduceNoise(normalizeLighting(gray, config), config), config);
    edgeMask = edge(enhanced, "Canny");
    [angle, angleConfidence] = localEstimateSkew(edgeMask, config);

    if angleConfidence >= config.rectification.minAngleConfidence && abs(angle) > 0.5
        rectifiedPlate = imrotate(cropped, angle, "bilinear", "crop");
        rotatedGray = imrotate(gray, angle, "bilinear", "crop");
        rotatedEnhanced = imrotate(enhanced, angle, "bilinear", "crop");
    else
        angle = 0;
        rectifiedPlate = cropped;
        rotatedGray = gray;
        rotatedEnhanced = enhanced;
    end

    rotatedBinary = imbinarize(rotatedEnhanced, "adaptive", "ForegroundPolarity", "dark");
    textMeta = localExtractTextRegion(rectifiedPlate, rotatedGray, rotatedEnhanced, config, hints);

    metadata = struct( ...
        "angle", angle, ...
        "angleConfidence", angleConfidence, ...
        "croppedPlate", cropped, ...
        "binaryMask", rotatedBinary, ...
        "layoutHint", string(textMeta.layoutHint), ...
        "textOnlyPlate", textMeta.textOnlyPlate, ...
        "textOnlyBBox", textMeta.textOnlyBBox, ...
        "rowBBoxes", textMeta.rowBBoxes, ...
        "rowImages", {textMeta.rowImages}, ...
        "rowCompositePlate", textMeta.rowCompositePlate, ...
        "textMask", textMeta.textMask, ...
        "rowCountEstimate", textMeta.rowCountEstimate);
end

function metadata = localEmptyMetadata()
    metadata = struct( ...
        "angle", 0, ...
        "angleConfidence", 0, ...
        "croppedPlate", [], ...
        "binaryMask", [], ...
        "layoutHint", "unknown", ...
        "textOnlyPlate", [], ...
        "textOnlyBBox", zeros(0, 4), ...
        "rowBBoxes", zeros(0, 4), ...
        "rowImages", {cell(0, 1)}, ...
        "rowCompositePlate", [], ...
        "textMask", [], ...
        "rowCountEstimate", 0);
end

function textMeta = localExtractTextRegion(rectifiedPlate, rectifiedGray, rectifiedEnhanced, config, hints)
    [brightMask, darkMask] = localBuildTextMasks(rectifiedEnhanced);
    brightMeta = localDescribeTextMask(brightMask, rectifiedGray, config, hints);
    darkMeta = localDescribeTextMask(darkMask, rectifiedGray, config, hints);

    if brightMeta.score >= darkMeta.score
        selectedMeta = brightMeta;
    else
        selectedMeta = darkMeta;
    end

    if isempty(selectedMeta.textOnlyBBox)
        fallbackBBox = [1 1 size(rectifiedGray, 2) size(rectifiedGray, 1)];
        textOnlyPlate = localUpscaleIfNeeded(rectifiedPlate, config);
        textMeta = struct( ...
            "layoutHint", "unknown", ...
            "textOnlyPlate", textOnlyPlate, ...
            "textOnlyBBox", fallbackBBox, ...
            "rowBBoxes", zeros(0, 4), ...
            "rowImages", {cell(0, 1)}, ...
            "rowCompositePlate", [], ...
            "textMask", selectedMeta.textMask, ...
            "rowCountEstimate", 0);
        return;
    end

    textOnlyPlate = imcrop(rectifiedPlate, selectedMeta.textOnlyBBox);
    rowCompositePlate = [];
    rowImages = cell(0, 1);
    if ~isempty(selectedMeta.rowBBoxes) && size(selectedMeta.rowBBoxes, 1) >= 2
        [rowCompositePlate, rowImages] = localBuildRowCompositePlate(rectifiedGray, selectedMeta.rowBBoxes);
    end

    textMeta = struct( ...
        "layoutHint", string(selectedMeta.layoutHint), ...
        "textOnlyPlate", localUpscaleIfNeeded(textOnlyPlate, config), ...
        "textOnlyBBox", selectedMeta.textOnlyBBox, ...
        "rowBBoxes", selectedMeta.rowBBoxes, ...
        "rowImages", {localUpscaleRowImages(rowImages, config)}, ...
        "rowCompositePlate", localUpscaleIfNeeded(rowCompositePlate, config), ...
        "textMask", selectedMeta.textMask, ...
        "rowCountEstimate", selectedMeta.rowCountEstimate);
end

function [brightMask, darkMask] = localBuildTextMasks(grayImage)
    brightMask = imbinarize(im2single(grayImage), "adaptive", ...
        "ForegroundPolarity", "bright", "Sensitivity", 0.44);
    darkMask = imbinarize(im2single(grayImage), "adaptive", ...
        "ForegroundPolarity", "dark", "Sensitivity", 0.44);

    brightMask = localCleanupTextMask(brightMask);
    darkMask = localCleanupTextMask(darkMask);
end

function mask = localCleanupTextMask(mask)
    mask = logical(mask);
    mask = imopen(mask, strel("rectangle", [2 2]));
    mask = imclose(mask, strel("rectangle", [2 3]));
    mask = bwareaopen(mask, max(2, round(numel(mask) * 0.0010)));
end

function meta = localDescribeTextMask(textMask, referenceGray, config, hints)
    meta = struct( ...
        "score", 0, ...
        "layoutHint", "unknown", ...
        "textOnlyBBox", [], ...
        "rowBBoxes", zeros(0, 4), ...
        "textMask", textMask, ...
        "rowCountEstimate", 0);
    twoRowProjectionMinScore = localRectificationRatio( ...
        config.rectification, "twoRowProjectionMinScore", 0.50);

    boxes = localCollectBoxes(textMask, config);
    if isempty(boxes)
        return;
    end

    boxes = localSelectBestTextCluster(boxes, textMask, hints);
    if isempty(boxes)
        return;
    end

    [rowGroups, hasTwoRows] = localSplitRows(boxes, textMask, config, hints);
    if hasTwoRows
        directTwoRowScore = localTwoRowMaskScore(rowGroups, boxes, size(textMask));
        if directTwoRowScore >= twoRowProjectionMinScore
            rowBBoxes = localRowBoundingBoxes(rowGroups, size(textMask), config);
            layoutHint = "two_row";
            rowCountEstimate = 2;
            layoutScore = directTwoRowScore;
        else
            hasTwoRows = false;
        end
    end

    if ~hasTwoRows
        [projectionRowBBoxes, projectionScore] = localProjectionRowBBoxes(textMask, boxes, config, hints);
        if size(projectionRowBBoxes, 1) >= 2 && ...
                projectionScore >= twoRowProjectionMinScore
            rowBBoxes = projectionRowBBoxes;
            layoutHint = "two_row";
            rowCountEstimate = 2;
            layoutScore = projectionScore;
        else
            rowBBoxes = localRowBoundingBoxes({boxes}, size(textMask), config);
            layoutHint = "single_line";
            rowCountEstimate = 1;
            layoutScore = localSingleRowMaskScore(boxes, size(textMask, 1));
        end
    end

    textOnlyBBox = localExpandTextBoundingBox(localUnionBox(boxes), size(textMask), config.rectification);
    coverageScore = localCoverageScore(boxes, size(textMask));
    contrastScore = localMaskContrast(referenceGray, textMask);
    meta.score = max(0, min(1, mean([layoutScore coverageScore contrastScore])));
    meta.layoutHint = layoutHint;
    meta.textOnlyBBox = textOnlyBBox;
    meta.rowBBoxes = rowBBoxes;
    meta.rowCountEstimate = rowCountEstimate;
end

function selectedBoxes = localSelectBestTextCluster(boxes, textMask, hints)
    selectedBoxes = boxes;
    if size(boxes, 1) < 3
        return;
    end

    if ~localExpectedTwoRow(hints)
        return;
    end

    [clusters, clusterIndex] = localHorizontalClusters(boxes); %#ok<ASGLU>
    if numel(clusters) <= 1
        return;
    end

    bestScore = -inf;
    bestCluster = boxes;

    for i = 1:numel(clusters)
        clusterBoxes = clusters{i};
        clusterBox = localUnionBox(clusterBoxes);
        widthRatio = clusterBox(3) / max(size(textMask, 2), 1);
        heightRatio = clusterBox(4) / max(size(textMask, 1), 1);
        areaScore = min(1, sum(clusterBoxes(:, 3) .* clusterBoxes(:, 4)) / ...
            max(clusterBox(3) * clusterBox(4), 1));
        componentScore = min(1, size(clusterBoxes, 1) / 6);
        widthScore = max(0, 1 - abs(widthRatio - 0.30) / 0.32);
        heightScore = max(0, 1 - abs(heightRatio - 0.48) / 0.40);
        xCenter = clusterBox(1) + clusterBox(3) / 2;
        centerScore = max(0, 1 - abs(xCenter - size(textMask, 2) / 2) / max(0.55 * size(textMask, 2), 1));

        rowScore = 0.45;
        [rowGroups, hasTwoRows] = localProjectionRowSplit(clusterBoxes, textMask, struct(), struct("layoutHint", "two_row")); %#ok<ASGLU>
        if hasTwoRows
            rowScore = 1.0;
        else
            rowScore = 0.28;
        end

        clusterScore = 0.28 * componentScore + 0.24 * areaScore + 0.20 * widthScore + ...
            0.14 * heightScore + 0.08 * centerScore + 0.06 * rowScore;
        if clusterScore > bestScore
            bestScore = clusterScore;
            bestCluster = clusterBoxes;
        end
    end

    selectedBoxes = bestCluster;
end

function [clusters, clusterIndex] = localHorizontalClusters(boxes)
    clusters = {};
    clusterIndex = zeros(size(boxes, 1), 1);
    if isempty(boxes)
        return;
    end

    [~, order] = sort(boxes(:, 1), "ascend");
    sortedBoxes = boxes(order, :);
    sortedEnds = sortedBoxes(:, 1) + sortedBoxes(:, 3);
    medianWidth = median(sortedBoxes(:, 3));
    gapThreshold = max(4, 0.85 * medianWidth);

    currentCluster = sortedBoxes(1, :);
    currentIndices = order(1);
    currentEnd = sortedEnds(1);
    clusterCount = 0;

    for i = 2:size(sortedBoxes, 1)
        gap = sortedBoxes(i, 1) - currentEnd;
        if gap <= gapThreshold
            currentCluster(end + 1, :) = sortedBoxes(i, :); %#ok<AGROW>
            currentIndices(end + 1, 1) = order(i); %#ok<AGROW>
            currentEnd = max(currentEnd, sortedEnds(i));
        else
            clusterCount = clusterCount + 1;
            clusters{clusterCount, 1} = currentCluster; %#ok<AGROW>
            clusterIndex(currentIndices) = clusterCount;
            currentCluster = sortedBoxes(i, :);
            currentIndices = order(i);
            currentEnd = sortedEnds(i);
        end
    end

    clusterCount = clusterCount + 1;
    clusters{clusterCount, 1} = currentCluster;
    clusterIndex(currentIndices) = clusterCount;
end

function [rowBBoxes, score] = localProjectionRowBBoxes(textMask, boxes, config, hints)
    rowBBoxes = zeros(0, 4);
    score = 0;

    if isempty(boxes)
        return;
    end

    expectedTwoRow = localExpectedTwoRow(hints);
    unionBox = localUnionBox(boxes);
    roiMask = imcrop(textMask, unionBox);
    if isempty(roiMask)
        return;
    end

    rowProfile = mean(roiMask, 2);
    positiveRows = rowProfile(rowProfile > 0);
    if numel(positiveRows) < 2
        return;
    end

    smoothWindow = max(3, 2 * floor(size(roiMask, 1) / 14) + 1);
    smoothProfile = movmean(rowProfile, smoothWindow);
    positiveSmooth = smoothProfile(smoothProfile > 0);
    if isempty(positiveSmooth)
        return;
    end

    thresholdFactor = 0.44;
    if ~expectedTwoRow
        thresholdFactor = 0.50;
    end
    activeThreshold = max(0.02, thresholdFactor * mean(positiveSmooth));
    activeRows = smoothProfile >= activeThreshold;
    rowRuns = localLogicalRuns(activeRows);
    if size(rowRuns, 1) < 2
        return;
    end

    runHeights = rowRuns(:, 2) - rowRuns(:, 1) + 1;
    [~, order] = sort(runHeights, "descend");
    selectedRuns = sortrows(rowRuns(order(1:min(2, size(rowRuns, 1))), :), 1);
    if size(selectedRuns, 1) < 2
        return;
    end

    roiWidth = size(roiMask, 2);
    splitCenter = floor((selectedRuns(1, 2) + selectedRuns(2, 1)) / 2);
    padRows = max(1, round(double(config.rectification.textPaddingRatio) * size(roiMask, 1)));
    minRowHeight = max(8, round(0.24 * unionBox(4)));
    topY1 = max(1, selectedRuns(1, 1) - padRows);
    topY2 = max(topY1 + minRowHeight - 1, splitCenter);
    bottomY1 = min(size(roiMask, 1), max(topY2 + 1, splitCenter + 1 - padRows));
    bottomY2 = min(size(roiMask, 1), max(bottomY1 + minRowHeight - 1, selectedRuns(2, 2) + padRows));

    if bottomY2 <= bottomY1
        bottomY1 = min(size(roiMask, 1) - 1, max(topY2 + 1, selectedRuns(2, 1)));
        bottomY2 = min(size(roiMask, 1), max(bottomY1 + 1, selectedRuns(2, 2) + padRows));
    end

    rowBBoxes = [ ...
        max(1, unionBox(1)) max(1, unionBox(2) + topY1 - 1) unionBox(3) max(2, topY2 - topY1 + 1); ...
        max(1, unionBox(1)) max(1, unionBox(2) + bottomY1 - 1) unionBox(3) max(2, bottomY2 - bottomY1 + 1)];

    separation = selectedRuns(2, 1) - selectedRuns(1, 2);
    heightBalance = max(0, 1 - abs(runHeights(order(1)) - runHeights(order(2))) / ...
        max(sum(runHeights(order(1:2))), 1));
    separationScore = min(1, separation / max(0.08 * size(roiMask, 1), 1));
    coverageScore = min(1, unionBox(3) / max(size(textMask, 2) * 0.10, 1));
    score = max(0, min(1, mean([heightBalance separationScore coverageScore])));
end

function boxes = localCollectBoxes(mask, config)
    stats = regionprops("table", mask, "BoundingBox", "Area");
    boxes = zeros(0, 4);
    if isempty(stats)
        return;
    end

    imageHeight = size(mask, 1);
    imageWidth = size(mask, 2);
    imageArea = numel(mask);
    minAreaRatio = double(config.rectification.rowComponentMinAreaRatio);

    for i = 1:height(stats)
        box = stats.BoundingBox(i, :);
        areaRatio = stats.Area(i) / max(imageArea, eps);
        heightRatio = box(4) / max(imageHeight, eps);
        widthRatio = box(3) / max(imageWidth, eps);
        aspectRatio = box(3) / max(box(4), eps);

        if areaRatio >= minAreaRatio && areaRatio <= 0.20 && ...
                heightRatio >= 0.08 && heightRatio <= 0.95 && ...
                widthRatio >= 0.008 && widthRatio <= 0.32 && ...
                aspectRatio >= 0.05 && aspectRatio <= 1.80
            boxes(end + 1, :) = box; %#ok<AGROW>
        end
    end
end

function [rowGroups, hasTwoRows] = localSplitRows(boxes, textMask, config, hints)
    rowGroups = {boxes};
    hasTwoRows = false;
    imageHeight = size(textMask, 1);

    if size(boxes, 1) < 4
        [rowGroups, hasTwoRows] = localProjectionRowSplit(boxes, textMask, config, hints);
        return;
    end

    centersY = boxes(:, 2) + boxes(:, 4) / 2;
    [sortedCenters, order] = sort(centersY, "ascend");
    sortedBoxes = boxes(order, :);
    gaps = diff(sortedCenters);
    if isempty(gaps)
        return;
    end

    [largestGap, splitIdx] = max(gaps);
    meanHeight = mean(sortedBoxes(:, 4));
    gapThreshold = max( ...
        double(config.rectification.rowSplitMinGapRatio) * imageHeight, ...
        0.45 * meanHeight);
    if largestGap < gapThreshold
        [rowGroups, hasTwoRows] = localProjectionRowSplit(boxes, textMask, config, hints);
        return;
    end

    topBoxes = sortedBoxes(1:splitIdx, :);
    bottomBoxes = sortedBoxes(splitIdx + 1:end, :);
    if size(topBoxes, 1) < 2 || size(bottomBoxes, 1) < 2
        [rowGroups, hasTwoRows] = localProjectionRowSplit(boxes, textMask, config, hints);
        return;
    end

    rowGroups = {topBoxes, bottomBoxes};
    hasTwoRows = true;
end

function [rowGroups, hasTwoRows] = localProjectionRowSplit(boxes, textMask, config, hints)
    rowGroups = {boxes};
    hasTwoRows = false;

    expectedTwoRow = localExpectedTwoRow(hints);
    if isempty(boxes) || (~expectedTwoRow && size(boxes, 1) < 4)
        return;
    end

    unionBox = localUnionBox(boxes);
    roiMask = imcrop(textMask, unionBox);
    if isempty(roiMask)
        return;
    end

    rowProfile = mean(roiMask, 2);
    positiveRows = rowProfile(rowProfile > 0);
    if numel(positiveRows) < 2
        return;
    end

    smoothWindow = max(3, 2 * floor(size(roiMask, 1) / 14) + 1);
    smoothProfile = movmean(rowProfile, smoothWindow);
    positiveSmooth = smoothProfile(smoothProfile > 0);
    if isempty(positiveSmooth)
        return;
    end

    activeThreshold = max(0.02, 0.48 * mean(positiveSmooth));
    activeRows = smoothProfile >= activeThreshold;
    rowRuns = localLogicalRuns(activeRows);
    if size(rowRuns, 1) < 2
        return;
    end

    runHeights = rowRuns(:, 2) - rowRuns(:, 1) + 1;
    [~, order] = sort(runHeights, "descend");
    selectedRuns = sortrows(rowRuns(order(1:min(2, size(rowRuns, 1))), :), 1);
    if size(selectedRuns, 1) < 2
        return;
    end

    splitRow = floor((selectedRuns(1, 2) + selectedRuns(2, 1)) / 2) + unionBox(2) - 1;
    topBoxes = boxes((boxes(:, 2) + boxes(:, 4) / 2) <= splitRow, :);
    bottomBoxes = boxes((boxes(:, 2) + boxes(:, 4) / 2) > splitRow, :);
    if isempty(topBoxes) || isempty(bottomBoxes)
        return;
    end

    if ~expectedTwoRow && (size(topBoxes, 1) < 2 || size(bottomBoxes, 1) < 2)
        return;
    end

    rowGroups = {topBoxes, bottomBoxes};
    hasTwoRows = true;
end

function isExpected = localExpectedTwoRow(hints)
    isExpected = false;
    if ~isstruct(hints)
        return;
    end

    if isfield(hints, "layoutHint") && string(hints.layoutHint) == "two_row"
        isExpected = true;
    elseif isfield(hints, "profileName") && string(hints.profileName) == "two_row"
        isExpected = true;
    end
end

function rowBBoxes = localRowBoundingBoxes(rowGroups, imageSize, config)
    rowBBoxes = zeros(numel(rowGroups), 4);
    for i = 1:numel(rowGroups)
        rowBBoxes(i, :) = localExpandBoundingBox( ...
            localUnionBox(rowGroups{i}), imageSize, 0.04 + 0.5 * double(config.rectification.textPaddingRatio));
    end
end

function score = localSingleRowMaskScore(boxes, imageHeight)
    componentCount = size(boxes, 1);
    densityScore = localExpectedCountScore(componentCount, [3 8]);
    centersY = boxes(:, 2) + boxes(:, 4) / 2;
    heightValues = boxes(:, 4);
    centerScore = max(0, 1 - std(centersY) / max(0.14 * imageHeight, eps));
    heightScore = max(0, 1 - std(heightValues) / max(mean(heightValues), eps));
    gapScore = localGapConsistency(boxes);
    score = max(0, min(1, mean([densityScore centerScore heightScore gapScore])));
end

function score = localTwoRowMaskScore(rowGroups, boxes, imageSize)
    rowCounts = [size(rowGroups{1}, 1) size(rowGroups{2}, 1)];
    countScore = localExpectedCountScore(size(boxes, 1), [4 9]);
    balanceScore = max(0, 1 - abs(diff(rowCounts)) / max(sum(rowCounts), 1));
    topCompactness = localRowCompactness(rowGroups{1}, imageSize(1));
    bottomCompactness = localRowCompactness(rowGroups{2}, imageSize(1));
    rowHeightScore = max(0, 1 - abs(mean(rowGroups{1}(:, 4)) - mean(rowGroups{2}(:, 4))) / ...
        max(mean(boxes(:, 4)), eps));
    horizontalConsistency = localTwoRowHorizontalConsistency(rowGroups, boxes);
    score = max(0, min(1, mean([ ...
        countScore ...
        balanceScore ...
        topCompactness ...
        bottomCompactness ...
        rowHeightScore ...
        horizontalConsistency])));
end

function score = localRowCompactness(boxes, imageHeight)
    centersY = boxes(:, 2) + boxes(:, 4) / 2;
    heightValues = boxes(:, 4);
    centerScore = max(0, 1 - std(centersY) / max(0.10 * imageHeight, eps));
    heightScore = max(0, 1 - std(heightValues) / max(mean(heightValues), eps));
    gapScore = localGapConsistency(boxes);
    score = max(0, min(1, mean([centerScore heightScore gapScore])));
end

function score = localTwoRowHorizontalConsistency(rowGroups, allBoxes)
topBox = localUnionBox(rowGroups{1});
bottomBox = localUnionBox(rowGroups{2});
allBox = localUnionBox(allBoxes);

topX2 = topBox(1) + topBox(3);
bottomX2 = bottomBox(1) + bottomBox(3);
overlapWidth = max(0, min(topX2, bottomX2) - max(topBox(1), bottomBox(1)));
overlapScore = overlapWidth / max(min(topBox(3), bottomBox(3)), eps);

centerTop = topBox(1) + topBox(3) / 2;
centerBottom = bottomBox(1) + bottomBox(3) / 2;
centerOffsetScore = max(0, 1 - abs(centerTop - centerBottom) / max(0.22 * allBox(3), eps));

widthBalance = max(0, 1 - abs(topBox(3) - bottomBox(3)) / max(max(topBox(3), bottomBox(3)), eps));
coverageScore = min([1, topBox(3) / max(0.42 * allBox(3), eps), bottomBox(3) / max(0.42 * allBox(3), eps)]);
score = max(0, min(1, mean([overlapScore centerOffsetScore widthBalance coverageScore])));
end

function score = localCoverageScore(boxes, imageSize)
    unionBox = localUnionBox(boxes);
    widthRatio = unionBox(3) / max(imageSize(2), eps);
    heightRatio = unionBox(4) / max(imageSize(1), eps);
    widthScore = max(0, 1 - abs(widthRatio - 0.68) / 0.42);
    heightScore = max(0, 1 - abs(heightRatio - 0.42) / 0.28);
    score = max(0, min(1, mean([widthScore heightScore])));
end

function score = localMaskContrast(grayImage, mask)
    if isempty(grayImage) || ~any(mask(:)) || all(mask(:))
        score = 0;
        return;
    end

    grayImage = im2double(grayImage);
    foregroundValues = grayImage(mask);
    backgroundValues = grayImage(~mask);
    score = min(1, abs(mean(foregroundValues) - mean(backgroundValues)) / 0.26);
end

function box = localUnionBox(boxes)
    x1 = floor(min(boxes(:, 1)));
    y1 = floor(min(boxes(:, 2)));
    x2 = ceil(max(boxes(:, 1) + boxes(:, 3)));
    y2 = ceil(max(boxes(:, 2) + boxes(:, 4)));
    box = [x1 y1 max(1, x2 - x1) max(1, y2 - y1)];
end

function expandedBox = localExpandBoundingBox(box, imageSize, paddingRatio)
    if isempty(box)
        expandedBox = [];
        return;
    end

    paddingX = box(3) * paddingRatio;
    paddingY = box(4) * paddingRatio;
    x1 = max(1, floor(box(1) - paddingX));
    y1 = max(1, floor(box(2) - paddingY));
    x2 = min(imageSize(2), ceil(box(1) + box(3) + paddingX));
    y2 = min(imageSize(1), ceil(box(2) + box(4) + paddingY));
    expandedBox = [x1 y1 max(1, x2 - x1) max(1, y2 - y1)];
end

function expandedBox = localExpandTextBoundingBox(box, imageSize, rectificationConfig)
    if isempty(box)
        expandedBox = [];
        return;
    end

    leftRatio = localRectificationRatio(rectificationConfig, "textPaddingLeftRatio", rectificationConfig.textPaddingRatio);
    rightRatio = localRectificationRatio(rectificationConfig, "textPaddingRightRatio", rectificationConfig.textPaddingRatio);
    verticalRatio = localRectificationRatio(rectificationConfig, "textPaddingVerticalRatio", rectificationConfig.textPaddingRatio);

    paddingLeft = box(3) * leftRatio;
    paddingRight = box(3) * rightRatio;
    paddingY = box(4) * verticalRatio;
    x1 = max(1, floor(box(1) - paddingLeft));
    y1 = max(1, floor(box(2) - paddingY));
    x2 = min(imageSize(2), ceil(box(1) + box(3) + paddingRight));
    y2 = min(imageSize(1), ceil(box(2) + box(4) + paddingY));
    expandedBox = [x1 y1 max(1, x2 - x1) max(1, y2 - y1)];
end

function ratio = localRectificationRatio(rectificationConfig, fieldName, fallbackValue)
    ratio = fallbackValue;
    if isstruct(rectificationConfig) && isfield(rectificationConfig, fieldName)
        ratio = double(rectificationConfig.(fieldName));
    end
end

function [rowCompositePlate, rowImages] = localBuildRowCompositePlate(grayImage, rowBBoxes)
    if isempty(rowBBoxes) || size(rowBBoxes, 1) < 2
        rowCompositePlate = [];
        rowImages = cell(0, 1);
        return;
    end

    rowImages = cell(size(rowBBoxes, 1), 1);
    targetHeight = 0;
    for i = 1:size(rowBBoxes, 1)
        rowImages{i} = imcrop(grayImage, rowBBoxes(i, :));
        if isempty(rowImages{i})
            rowCompositePlate = [];
            rowImages = cell(0, 1);
            return;
        end
        rowImages{i} = localTrimRowImage(rowImages{i});
        targetHeight = max(targetHeight, size(rowImages{i}, 1));
    end

    spacerWidth = max(4, round(0.08 * targetHeight));
    for i = 1:numel(rowImages)
        resized = localResizeToHeight(rowImages{i}, targetHeight);
        rowImages{i} = resized;
        if i == 1
            rowCompositePlate = resized;
        else
            spacer = localWhiteSpacer(targetHeight, spacerWidth, resized);
            rowCompositePlate = cat(2, rowCompositePlate, spacer, resized);
        end
    end
end

function resizedImage = localResizeToHeight(imageIn, targetHeight)
    if isempty(imageIn)
        resizedImage = imageIn;
        return;
    end

    if size(imageIn, 1) == targetHeight
        resizedImage = imageIn;
        return;
    end

    targetWidth = max(1, round(size(imageIn, 2) * targetHeight / max(size(imageIn, 1), 1)));
    resizedImage = imresize(imageIn, [targetHeight targetWidth], "bicubic");
end

function spacer = localWhiteSpacer(targetHeight, spacerWidth, referenceImage)
    if ndims(referenceImage) == 3
        spacer = uint8(255 * ones(targetHeight, spacerWidth, size(referenceImage, 3)));
    else
        spacer = uint8(255 * ones(targetHeight, spacerWidth));
    end
end

function trimmedImage = localTrimRowImage(rowImage)
    trimmedImage = rowImage;
    if isempty(rowImage)
        return;
    end

    grayImage = rowImage;
    if ndims(grayImage) == 3
        grayImage = im2gray(grayImage);
    end

    grayImage = im2single(grayImage);
    brightMask = imbinarize(grayImage, "adaptive", "ForegroundPolarity", "bright", "Sensitivity", 0.42);
    darkMask = imbinarize(grayImage, "adaptive", "ForegroundPolarity", "dark", "Sensitivity", 0.42);
    textMask = bwareaopen(brightMask | darkMask, max(2, round(numel(grayImage) * 0.0015)));
    cropBox = localBestRowCropBox(textMask);
    if isempty(cropBox)
        return;
    end

    trimmedImage = imcrop(rowImage, cropBox);
end

function cropBox = localBestRowCropBox(textMask)
    cropBox = [];
    if isempty(textMask) || ~any(textMask(:))
        return;
    end

    textMask = logical(textMask);
    columnProfile = mean(textMask, 1);
    positiveColumns = columnProfile(columnProfile > 0);
    if isempty(positiveColumns)
        return;
    end

    activeColumns = columnProfile >= max(0.02, 0.40 * mean(positiveColumns));
    columnRuns = localLogicalRuns(activeColumns);
    if isempty(columnRuns)
        return;
    end
    bestScore = -inf;
    bestBox = [];
    imageHeight = size(textMask, 1);
    imageWidth = size(textMask, 2);
    columnRuns = localMergeRuns(columnRuns, max(2, round(0.08 * imageWidth)));
    stats = regionprops("table", textMask, "BoundingBox", "Area");

    for i = 1:size(columnRuns, 1)
        x1 = columnRuns(i, 1);
        x2 = columnRuns(i, 2);
        if (x2 - x1 + 1) < 3
            continue;
        end

        runMask = textMask(:, x1:x2);
        if ~any(runMask(:))
            continue;
        end

        rowProfile = mean(runMask, 2);
        activeRows = rowProfile >= max(0.02, 0.35 * mean(rowProfile(rowProfile > 0)));
        rowRuns = localLogicalRuns(activeRows);
        if isempty(rowRuns)
            continue;
        end

        y1 = rowRuns(1, 1);
        y2 = rowRuns(end, 2);
        runWidth = x2 - x1 + 1;
        runHeight = y2 - y1 + 1;
        widthRatio = runWidth / max(imageWidth, 1);
        heightRatio = runHeight / max(imageHeight, 1);
        density = nnz(runMask(y1:y2, :)) / max(runWidth * runHeight, 1);

        componentCount = 0;
        for j = 1:height(stats)
            box = stats.BoundingBox(j, :);
            boxCenterX = box(1) + box(3) / 2;
            if boxCenterX >= x1 && boxCenterX <= x2
                componentCount = componentCount + 1;
            end
        end

        widthScore = max(0, 1 - abs(widthRatio - 0.55) / 0.45);
        heightScore = max(0, 1 - abs(heightRatio - 0.58) / 0.40);
        densityScore = min(1, density / 0.42);
        componentScore = min(1, componentCount / 4);
        edgePenalty = 0.10 * double(x1 == 1 || x2 == imageWidth);
        score = 0.34 * componentScore + 0.24 * widthScore + 0.20 * heightScore + ...
            0.22 * densityScore - edgePenalty;

        if score > bestScore
            bestScore = score;
            bestBox = [x1 y1 runWidth runHeight];
        end
    end

    if isempty(bestBox)
        return;
    end

    paddingX = max(1, round(0.04 * imageWidth));
    paddingY = max(1, round(0.08 * imageHeight));
    x1 = max(1, bestBox(1) - paddingX);
    y1 = max(1, bestBox(2) - paddingY);
    x2 = min(imageWidth, bestBox(1) + bestBox(3) - 1 + paddingX);
    y2 = min(imageHeight, bestBox(2) + bestBox(4) - 1 + paddingY);
    cropBox = [x1 y1 max(2, x2 - x1 + 1) max(2, y2 - y1 + 1)];
end

function mergedRuns = localMergeRuns(runs, maxGap)
    mergedRuns = runs;
    if size(runs, 1) <= 1
        return;
    end

    mergedRuns = runs(1, :);
    for i = 2:size(runs, 1)
        gap = runs(i, 1) - mergedRuns(end, 2) - 1;
        if gap <= maxGap
            mergedRuns(end, 2) = runs(i, 2);
        else
            mergedRuns(end + 1, :) = runs(i, :); %#ok<AGROW>
        end
    end
end

function rowImages = localUpscaleRowImages(rowImages, config)
    for i = 1:numel(rowImages)
        rowImages{i} = localUpscaleIfNeeded(rowImages{i}, config);
    end
end

function outputImage = localUpscaleIfNeeded(inputImage, config)
    outputImage = inputImage;
    if isempty(inputImage)
        return;
    end

    imageHeight = size(inputImage, 1);
    if imageHeight <= 0
        return;
    end

    minTextHeight = double(config.rectification.minTextHeightPixels);
    maxScaleFactor = double(config.rectification.maxTextUpscaleFactor);
    if imageHeight >= minTextHeight
        return;
    end

    scaleFactor = min(maxScaleFactor, minTextHeight / imageHeight);
    if scaleFactor > 1.01
        outputImage = imresize(inputImage, scaleFactor, "bicubic");
    end
end

function [angle, confidence] = localEstimateSkew(edgeMask, config)
    % LOCAL_ESTIMATESKEW Estimate the skew angle of a plate candidate using Hough transform on edge mask.

    angle = 0;
    confidence = 0;

    if nnz(edgeMask) == 0
        return;
    end

    [H, theta, rho] = hough(edgeMask); %#ok<ASGLU>
    peakCount = min(8, max(1, round(nnz(edgeMask) / 250)));
    peaks = houghpeaks(H, peakCount, "Threshold", ceil(0.25 * max(H(:))));
    if isempty(peaks)
        return;
    end

    lines = houghlines(edgeMask, theta, rho, peaks, ...
        "FillGap", 8, ...
        "MinLength", config.rectification.minLineLength);
    if isempty(lines)
        return;
    end

    candidateAngles = [];
    candidateWeights = [];
    for i = 1:numel(lines)
        point1 = double(lines(i).point1);
        point2 = double(lines(i).point2);
        delta = point2 - point1;
        lineLength = hypot(delta(1), delta(2));
        if lineLength < config.rectification.minLineLength
            continue;
        end

        lineAngle = atan2d(delta(2), delta(1));
        lineAngle = mod(lineAngle + 90, 180) - 90;
        if abs(lineAngle) > config.rectification.maxRotationDegrees
            continue;
        end

        candidateAngles(end+1) = lineAngle; %#ok<AGROW>
        candidateWeights(end+1) = lineLength; %#ok<AGROW>
    end

    if isempty(candidateAngles)
        return;
    end

    angle = sum(candidateAngles .* candidateWeights) / sum(candidateWeights);
    inlierMask = abs(candidateAngles - angle) <= 2.5;
    confidence = sum(candidateWeights(inlierMask)) / max(sum(candidateWeights), eps);
    angle = max(min(angle, config.rectification.maxRotationDegrees), -config.rectification.maxRotationDegrees);
end

function score = localExpectedCountScore(count, targetRange)
    if count >= targetRange(1) && count <= targetRange(2)
        score = 1;
    elseif count <= 1
        score = 0;
    else
        distance = min(abs(count - targetRange(1)), abs(count - targetRange(2)));
        score = max(0.15, 0.85 - 0.12 * distance);
    end
end

function score = localGapConsistency(boxes)
    if size(boxes, 1) <= 2
        score = 0.75;
        return;
    end

    [~, order] = sort(boxes(:, 1), "ascend");
    sortedBoxes = boxes(order, :);
    gaps = sortedBoxes(2:end, 1) - (sortedBoxes(1:end-1, 1) + sortedBoxes(1:end-1, 3));
    gaps = gaps(gaps >= 0);

    if isempty(gaps)
        score = 0.25;
        return;
    end

    score = max(0, 1 - std(gaps) / max(mean(gaps) + 1, eps));
    score = min(score, 1);
end

function runs = localLogicalRuns(maskVector)
    maskVector = logical(maskVector(:));
    runs = zeros(0, 2);
    if ~any(maskVector)
        return;
    end

    changes = diff([false; maskVector; false]);
    starts = find(changes == 1);
    ends = find(changes == -1) - 1;
    runs = [starts ends];
end

function paddedBBox = localPadBBox(bbox, imageSize, paddingRatio)
    % LOCALPADBBOX Expand the bounding box by a certain padding ratio while ensuring it stays within image bounds.

    paddingX = bbox(3) * paddingRatio;
    paddingY = bbox(4) * paddingRatio;

    x1 = max(1, floor(bbox(1) - paddingX));
    y1 = max(1, floor(bbox(2) - paddingY));
    x2 = min(imageSize(2), ceil(bbox(1) + bbox(3) + paddingX));
    y2 = min(imageSize(1), ceil(bbox(2) + bbox(4) + paddingY));

    paddedBBox = [x1 y1 max(1, x2 - x1) max(1, y2 - y1)];
end
