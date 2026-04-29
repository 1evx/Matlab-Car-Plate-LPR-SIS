function [characterImages, characterBBoxes, metadata] = segmentCharacters(plateImage, config)
    %SEGMENTCHARACTERS Segment likely character regions from a rectified plate crop.

    config = validateConfig(config);
    gray = im2gray(plateImage);
    roiBox = localInnerPlateBox(size(gray), config);
    grayRoi = imcrop(gray, roiBox);
    if isempty(grayRoi)
        grayRoi = gray;
        roiBox = [1 1 size(gray, 2) size(gray, 1)];
    end

    normalized = normalizeLighting(grayRoi, config);
    denoised = reduceNoise(normalized, config);
    enhanced = enhanceContrast(denoised, config);

    maskCandidates = localBuildMaskCandidates(enhanced, config);
    [bestMask, polarityUsed, candidateSummaries, selectedMaskScore, roiCharacterBoxes, rawComponentCount, rowLayout, qualityScore, qualityBreakdown] = ...
        localSelectBestMask(maskCandidates, config);
    characterBBoxes = localMapBoxesToPlate(roiCharacterBoxes, roiBox);
    characterCount = size(roiCharacterBoxes, 1);
    characterCount = min(characterCount, config.segmentation.maxCharacters);
    roiCharacterBoxes = roiCharacterBoxes(1:characterCount, :);
    characterBBoxes = characterBBoxes(1:characterCount, :);

    characterImages = cell(1, characterCount);
    overlay = plateImage;

    for i = 1:characterCount
        charBox = ceil(roiCharacterBoxes(i, :));
        charCrop = imcrop(bestMask, charBox);
        charCrop = localTightCrop(charCrop);
        charCrop = padarray(charCrop, [2 2], 0, "both");
        characterImages{i} = charCrop;
        overlay = drawBoundingBoxes(overlay, characterBBoxes(i, :), [0 255 255], 2);
    end

    metadata = struct( ...
        "grayImage", gray, ...
        "roiBox", roiBox, ...
        "normalizedImage", normalized, ...
        "enhancedImage", enhanced, ...
        "binaryMask", bestMask, ...
        "overlay", overlay, ...
        "polarityUsed", polarityUsed, ...
        "selectedMaskScore", selectedMaskScore, ...
        "maskCandidates", {candidateSummaries}, ...
        "rawComponentCount", rawComponentCount, ...
        "orderedBoxes", characterBBoxes, ...
        "rowLayout", rowLayout, ...
        "qualityScore", qualityScore, ...
        "qualityBreakdown", qualityBreakdown);
end

function roiBox = localInnerPlateBox(imageSize, config)
    plateHeight = imageSize(1);
    plateWidth = imageSize(2);
    marginX = round(plateWidth * config.segmentation.innerCropRatioX);
    marginY = round(plateHeight * config.segmentation.innerCropRatioY);

    x1 = max(1, 1 + marginX);
    y1 = max(1, 1 + marginY);
    x2 = min(plateWidth, plateWidth - marginX);
    y2 = min(plateHeight, plateHeight - marginY);
    roiBox = [x1 y1 max(1, x2 - x1) max(1, y2 - y1)];
end

function mappedBoxes = localMapBoxesToPlate(boxes, roiBox)
    mappedBoxes = boxes;
    if isempty(mappedBoxes)
        return;
    end

    mappedBoxes(:, 1) = mappedBoxes(:, 1) + roiBox(1) - 1;
    mappedBoxes(:, 2) = mappedBoxes(:, 2) + roiBox(2) - 1;
end

function candidates = localBuildMaskCandidates(grayPlate, config)
    sensitivity = config.segmentation.adaptiveSensitivity;
    rawCandidates = { ...
        struct("name", "adaptive_bright", "mask", imbinarize(grayPlate, "adaptive", ...
            "ForegroundPolarity", "bright", "Sensitivity", sensitivity)), ...
        struct("name", "adaptive_dark", "mask", ~imbinarize(grayPlate, "adaptive", ...
            "ForegroundPolarity", "dark", "Sensitivity", sensitivity)), ...
        struct("name", "adaptive_bright_soft", "mask", imbinarize(grayPlate, "adaptive", ...
            "ForegroundPolarity", "bright", "Sensitivity", min(0.75, sensitivity + 0.10))), ...
        struct("name", "adaptive_dark_soft", "mask", ~imbinarize(grayPlate, "adaptive", ...
            "ForegroundPolarity", "dark", "Sensitivity", min(0.75, sensitivity + 0.10))), ...
        struct("name", "global_bright", "mask", imbinarize(grayPlate)), ...
        struct("name", "global_dark", "mask", ~imbinarize(grayPlate))};

    candidates = cell(1, numel(rawCandidates));
    for i = 1:numel(rawCandidates)
        candidates{i} = struct( ...
            "name", rawCandidates{i}.name, ...
            "mask", localCleanupMask(rawCandidates{i}.mask, config));
    end
end

function cleanedMask = localCleanupMask(binaryMask, config)
    cleanedMask = logical(binaryMask);
    cleanedMask = imopen(cleanedMask, strel("rectangle", config.segmentation.openKernel));
    cleanedMask = imclose(cleanedMask, strel("rectangle", config.segmentation.closeKernel));
    cleanedMask = bwareaopen(cleanedMask, config.segmentation.minComponentArea);
    cleanedMask = localRemoveHorizontalBorderComponents(cleanedMask, config);
end

function cleanedMask = localRemoveHorizontalBorderComponents(cleanedMask, config)
    stats = regionprops("table", cleanedMask, "BoundingBox", "Area");
    for i = 1:height(stats)
        box = stats.BoundingBox(i, :);
        widthRatio = box(3) / max(size(cleanedMask, 2), eps);
        heightRatio = box(4) / max(size(cleanedMask, 1), eps);
        touchesTop = box(2) <= 2;
        touchesBottom = (box(2) + box(4)) >= (size(cleanedMask, 1) - 1);
        isLongHorizontal = widthRatio >= config.segmentation.horizontalBorderWidthRatio && ...
            heightRatio <= config.segmentation.horizontalBorderHeightRatio;

        if (touchesTop || touchesBottom) && isLongHorizontal
            cleanedMask = localEraseBox(cleanedMask, box);
        end
    end
end

function mask = localEraseBox(mask, box)
    x1 = max(1, floor(box(1)));
    y1 = max(1, floor(box(2)));
    x2 = min(size(mask, 2), ceil(box(1) + box(3) - 1));
    y2 = min(size(mask, 1), ceil(box(2) + box(4) - 1));
    mask(y1:y2, x1:x2) = false;
end

function [bestMask, polarityUsed, candidateSummaries, selectedMaskScore, bestBoxes, bestRawComponentCount, bestRowLayout, bestQualityScore, bestQualityBreakdown] = ...
        localSelectBestMask(maskCandidates, config)

    bestMask = false(size(maskCandidates{1}.mask));
    polarityUsed = "none";
    bestScore = -inf;
    selectedMaskScore = 0;
    bestBoxes = zeros(0, 4);
    bestRawComponentCount = 0;
    bestRowLayout = "single_row";
    bestQualityScore = 0;
    bestQualityBreakdown = struct();
    candidateSummaries = struct("name", {}, "score", {}, "characterCount", {}, "foregroundRatio", {}, "layoutScore", {}, "countScore", {}, "sizeConsistency", {}, "projectionGroupCount", {}, "rowLayout", {});

    for i = 1:numel(maskCandidates)
        candidate = maskCandidates{i};
        [boxes, rawComponentCount, rowLayout, projectionGroupCount] = localExtractCharacterBoxes(candidate.mask, config);
        [layoutScore, qualityBreakdown] = localCharacterLayoutQuality(candidate.mask, boxes, rowLayout, config);
        foregroundRatio = nnz(candidate.mask) / max(numel(candidate.mask), 1);
        candidateCount = size(boxes, 1);
        countScore = localExpectedCountScore(candidateCount, ...
            [config.segmentation.minCharacters min(config.segmentation.maxCharacters, 8)]);
        if candidateCount < config.segmentation.minCharacters || candidateCount > config.segmentation.maxCharacters
            countScore = countScore * 0.25;
        end
        densityScore = 1 - min(abs(foregroundRatio - 0.20) / 0.20, 1);
        score = 0.55 * countScore + 0.35 * layoutScore + 0.10 * densityScore;

        candidateSummaries(end+1) = struct( ... %#ok<AGROW>
            "name", string(candidate.name), ...
            "score", score, ...
            "characterCount", candidateCount, ...
            "foregroundRatio", foregroundRatio, ...
            "layoutScore", layoutScore, ...
            "countScore", countScore, ...
            "sizeConsistency", qualityBreakdown.heightConsistency, ...
            "projectionGroupCount", projectionGroupCount, ...
            "rowLayout", rowLayout);

        if score > bestScore
            bestScore = score;
            bestMask = candidate.mask;
            polarityUsed = string(candidate.name);
            selectedMaskScore = score;
            bestBoxes = boxes;
            bestRawComponentCount = rawComponentCount;
            bestRowLayout = rowLayout;
            bestQualityScore = layoutScore;
            bestQualityBreakdown = qualityBreakdown;
        end
    end
end

function [characterBBoxes, rawComponentCount, rowLayout, projectionGroupCount] = localExtractCharacterBoxes(binaryMask, config)
    binaryMask = logical(binaryMask);
    plateHeight = size(binaryMask, 1);
    plateWidth = size(binaryMask, 2);
    plateArea = numel(binaryMask);

    stats = regionprops("table", binaryMask, "BoundingBox", "Area");
    rawComponentCount = height(stats);
    characterBBoxes = zeros(0, 4);
    rowLayout = "single_row";
    projectionGroupCount = 0;

    for i = 1:height(stats)
        rawBox = stats.BoundingBox(i, :);
        splitBoxes = localSplitWideComponent(binaryMask, rawBox, config);

        for j = 1:size(splitBoxes, 1)
            candidateBox = splitBoxes(j, :);
            componentArea = nnz(imcrop(binaryMask, ceil(candidateBox)));
            if localIsValidCharacterBox(candidateBox, componentArea, plateHeight, plateWidth, plateArea, config)
                characterBBoxes(end+1, :) = candidateBox; %#ok<AGROW>
            end
        end
    end

    projectionBoxes = localProjectionCharacterGroups(binaryMask, config);
    projectionGroupCount = size(projectionBoxes, 1);
    for i = 1:size(projectionBoxes, 1)
        candidateBox = projectionBoxes(i, :);
        componentArea = nnz(imcrop(binaryMask, ceil(candidateBox)));
        if localIsValidCharacterBox(candidateBox, componentArea, plateHeight, plateWidth, plateArea, config)
            characterBBoxes(end+1, :) = candidateBox; %#ok<AGROW>
        end
    end

    if isempty(characterBBoxes)
        return;
    end

    characterBBoxes = unique(round(characterBBoxes), "rows");
    characterBBoxes = localFilterOverlappingBoxes(characterBBoxes);
    [characterBBoxes, rowLayout] = localOrderCharacterBoxes(characterBBoxes, plateHeight, config);
end

function projectionBoxes = localProjectionCharacterGroups(binaryMask, config)
    projectionBoxes = zeros(0, 4);
    columnProfile = sum(binaryMask, 1) / max(size(binaryMask, 1), 1);
    activeColumns = columnProfile >= config.segmentation.projectionForegroundThreshold;
    activeColumns = imclose(activeColumns, ones(1, 3));
    activeColumns = logical(activeColumns);

    groups = bwconncomp(activeColumns);
    for i = 1:groups.NumObjects
        cols = groups.PixelIdxList{i};
        if isempty(cols)
            continue;
        end

        x1 = cols(1);
        x2 = cols(end);
        groupMask = binaryMask(:, x1:x2);
        props = regionprops(groupMask, "BoundingBox", "Area");
        if isempty(props)
            continue;
        end

        [~, idx] = max([props.Area]);
        tightBox = props(idx).BoundingBox;
        projectionBoxes(end+1, :) = [ ... %#ok<AGROW>
            x1 + tightBox(1) - 1, ...
            tightBox(2), ...
            tightBox(3), ...
            tightBox(4)];
    end
end

function [orderedBoxes, rowLayout] = localOrderCharacterBoxes(boxes, plateHeight, config)
    if size(boxes, 1) <= 1
        orderedBoxes = boxes;
        rowLayout = "single_row";
        return;
    end

    centersY = boxes(:, 2) + boxes(:, 4) / 2;
    centerSpread = (max(centersY) - min(centersY)) / max(plateHeight, eps);
    if centerSpread < config.segmentation.twoRowCenterGapRatio
        orderedBoxes = sortrows(boxes, 1);
        rowLayout = "single_row";
        return;
    end

    splitThreshold = median(centersY);
    topMask = centersY <= splitThreshold;
    bottomMask = ~topMask;

    topBoxes = sortrows(boxes(topMask, :), 1);
    bottomBoxes = sortrows(boxes(bottomMask, :), 1);
    orderedBoxes = [topBoxes; bottomBoxes];
    rowLayout = "two_row";
end

function splitBoxes = localSplitWideComponent(binaryMask, box, config)
    splitBoxes = box;

    plateWidth = size(binaryMask, 2);
    widthRatio = box(3) / max(plateWidth, eps);
    aspectRatio = box(3) / max(box(4), eps);
    shouldSplit = widthRatio >= config.segmentation.splitMinWidthRatio && ...
        aspectRatio >= config.segmentation.splitMinAspectRatio;

    if ~shouldSplit
        return;
    end

    componentMask = imcrop(binaryMask, box);
    if isempty(componentMask)
        return;
    end

    columnProfile = sum(componentMask, 1);
    if max(columnProfile) == 0
        return;
    end

    normalizedProfile = movmean(columnProfile, 3) / max(columnProfile);
    gapMask = normalizedProfile <= config.segmentation.splitMinForegroundDip;
    gapMask([1 end]) = false;
    gapComponents = bwconncomp(gapMask);

    if gapComponents.NumObjects == 0
        return;
    end

    gapCenters = zeros(1, gapComponents.NumObjects);
    for i = 1:gapComponents.NumObjects
        columns = gapComponents.PixelIdxList{i};
        if numel(columns) < config.segmentation.splitMinColumnGap
            continue;
        end
        gapCenters(i) = round(mean(columns));
    end

    gapCenters = unique(gapCenters(gapCenters > 1 & gapCenters < size(componentMask, 2)));
    if isempty(gapCenters)
        return;
    end

    segmentStarts = [1 gapCenters + 1];
    segmentEnds = [gapCenters size(componentMask, 2)];
    splitBoxes = zeros(0, 4);

    for i = 1:numel(segmentStarts)
        segmentMask = componentMask(:, segmentStarts(i):segmentEnds(i));
        props = regionprops(segmentMask, "BoundingBox", "Area");
        if isempty(props)
            continue;
        end

        [~, largestIdx] = max([props.Area]);
        tightBox = props(largestIdx).BoundingBox;
        mappedBox = [ ...
            box(1) + segmentStarts(i) + tightBox(1) - 2, ...
            box(2) + tightBox(2) - 1, ...
            tightBox(3), ...
            tightBox(4)];
        splitBoxes(end+1, :) = mappedBox; %#ok<AGROW>
    end

    if isempty(splitBoxes)
        splitBoxes = box;
    end
end

function isValid = localIsValidCharacterBox(box, componentArea, plateHeight, plateWidth, plateArea, config)
    heightRatio = box(4) / max(plateHeight, eps);
    widthRatio = box(3) / max(plateWidth, eps);
    areaRatio = componentArea / max(plateArea, eps);
    aspectRatio = box(3) / max(box(4), eps);

    isValid = areaRatio >= config.segmentation.minAreaRatio && ...
        areaRatio <= config.segmentation.maxAreaRatio && ...
        heightRatio >= config.segmentation.minHeightRatio && ...
        heightRatio <= config.segmentation.maxHeightRatio && ...
        widthRatio >= config.segmentation.minWidthRatio && ...
        widthRatio <= config.segmentation.maxWidthRatio && ...
        aspectRatio >= config.segmentation.minAspectRatio && ...
        aspectRatio <= config.segmentation.maxAspectRatio;

    if ~isValid || ~config.segmentation.rejectBorderTouchingTallComponents
        return;
    end

    touchesHorizontalBorder = box(1) <= 3 || (box(1) + box(3)) >= (plateWidth - 2);
    touchesVerticalBorder = box(2) <= 3 || (box(2) + box(4)) >= (plateHeight - 2);
    if touchesHorizontalBorder && touchesVerticalBorder && heightRatio >= 0.80
        isValid = false;
    end
end

function [qualityScore, qualityBreakdown] = localCharacterLayoutQuality(binaryMask, boxes, rowLayout, config)
    if isempty(boxes)
        qualityScore = 0;
        qualityBreakdown = struct( ...
            "count", 0, ...
            "heightConsistency", 0, ...
            "spacingConsistency", 0, ...
            "occupancy", 0, ...
            "rowBalance", 0, ...
            "final", 0);
        return;
    end

    expectedRange = [max(config.segmentation.minCharacters, 4) min(config.segmentation.maxCharacters, 8)];
    countScore = localExpectedCountScore(size(boxes, 1), expectedRange);

    heights = boxes(:, 4);
    heightConsistency = max(0, 1 - std(heights) / max(mean(heights), eps));

    occupancyScores = zeros(size(boxes, 1), 1);
    for i = 1:size(boxes, 1)
        crop = imcrop(binaryMask, ceil(boxes(i, :)));
        occupancy = nnz(crop) / max(numel(crop), 1);
        occupancyScores(i) = max(0, 1 - abs(occupancy - 0.42) / 0.42);
    end
    occupancyScore = mean(occupancyScores);
    [spacingConsistency, rowBalance] = localSpacingAndRowQuality(boxes, rowLayout);

    qualityScore = max(0, min(1, ...
        0.28 * countScore + ...
        0.22 * heightConsistency + ...
        0.22 * spacingConsistency + ...
        0.18 * occupancyScore + ...
        0.10 * rowBalance));
    qualityBreakdown = struct( ...
        "count", countScore, ...
        "heightConsistency", heightConsistency, ...
        "spacingConsistency", spacingConsistency, ...
        "occupancy", occupancyScore, ...
        "rowBalance", rowBalance, ...
        "final", qualityScore);
end

function score = localExpectedCountScore(count, expectedRange)
    if count >= expectedRange(1) && count <= expectedRange(2)
        score = 1;
    elseif count <= 1
        score = 0;
    else
        distance = min(abs(count - expectedRange(1)), abs(count - expectedRange(2)));
        score = max(0.10, 0.85 - 0.12 * distance);
    end
end

function [spacingConsistency, rowBalance] = localSpacingAndRowQuality(boxes, rowLayout)
    if size(boxes, 1) <= 1
        spacingConsistency = 0.20;
        rowBalance = strcmp(rowLayout, "single_row");
        return;
    end

    if rowLayout == "two_row"
        centersY = boxes(:, 2) + boxes(:, 4) / 2;
        splitY = median(centersY);
        topMask = centersY <= splitY;
        bottomMask = ~topMask;
        topScore = localRowGapScore(boxes(topMask, :));
        bottomScore = localRowGapScore(boxes(bottomMask, :));
        spacingConsistency = mean([topScore bottomScore]);
        topCount = nnz(topMask);
        bottomCount = nnz(bottomMask);
        rowBalance = max(0, 1 - abs(topCount - bottomCount) / max(topCount + bottomCount, 1));
    else
        spacingConsistency = localRowGapScore(boxes);
        rowBalance = 1;
    end
end

function score = localRowGapScore(boxes)
    if size(boxes, 1) <= 2
        score = 0.80;
        return;
    end

    [~, sortIdx] = sort(boxes(:, 1), "ascend");
    orderedBoxes = boxes(sortIdx, :);
    gaps = orderedBoxes(2:end, 1) - (orderedBoxes(1:end-1, 1) + orderedBoxes(1:end-1, 3));
    gaps = gaps(gaps >= 0);

    if isempty(gaps)
        score = 0.25;
        return;
    end

    score = max(0, 1 - std(gaps) / max(mean(gaps) + 1, eps));
    score = min(score, 1);
end

function filteredBoxes = localFilterOverlappingBoxes(boxes)
    if size(boxes, 1) <= 1
        filteredBoxes = boxes;
        return;
    end

    keepMask = true(size(boxes, 1), 1);
    areas = boxes(:, 3) .* boxes(:, 4);

    for i = 1:size(boxes, 1)
        if ~keepMask(i)
            continue;
        end

        for j = i + 1:size(boxes, 1)
            if ~keepMask(j)
                continue;
            end

            overlapArea = localOverlapOverMin(boxes(i, :), boxes(j, :));
            if overlapArea >= 0.70
                if areas(i) >= areas(j)
                    keepMask(j) = false;
                else
                    keepMask(i) = false;
                end
            end
        end
    end

    filteredBoxes = boxes(keepMask, :);
end

function overlapRatio = localOverlapOverMin(boxA, boxB)
    x1 = max(boxA(1), boxB(1));
    y1 = max(boxA(2), boxB(2));
    x2 = min(boxA(1) + boxA(3), boxB(1) + boxB(3));
    y2 = min(boxA(2) + boxA(4), boxB(2) + boxB(4));

    intersectionWidth = max(0, x2 - x1);
    intersectionHeight = max(0, y2 - y1);
    intersectionArea = intersectionWidth * intersectionHeight;
    minArea = min(boxA(3) * boxA(4), boxB(3) * boxB(4));

    if minArea == 0
        overlapRatio = 0;
    else
        overlapRatio = intersectionArea / minArea;
    end
end

function croppedMask = localTightCrop(binaryMask)
    croppedMask = logical(binaryMask);
    props = regionprops(croppedMask, "BoundingBox", "Area");
    if isempty(props)
        return;
    end

    [~, idx] = max([props.Area]);
    tightBox = ceil(props(idx).BoundingBox);
    croppedMask = imcrop(croppedMask, tightBox);
end
