function [bbox, metadata] = detectPlateRegion(grayImage, config)
    %DETECTPLATEREGION Detect likely plate candidates and rank them late.

    config = validateConfig(config);
    grayImage = im2uint8(grayImage);
    enhancedImage = enhanceContrast( ...
        reduceNoise(normalizeLighting(grayImage, config), config), ...
        config);
    imageSize = size(grayImage);

    candidateRecords = localEmptyCandidateRecords();
    branchMasks = localInitializeBranchMasks(imageSize);

    for scale = config.detection.scales
        scaledGray = imresize(grayImage, scale, "bilinear");
        scaledEnhanced = imresize(enhancedImage, scale, "bilinear");

        [edgeRecords, edgeDebug] = localGenerateEdgeCandidates( ...
            scaledGray, scaledEnhanced, grayImage, config, scale, false);
        [priorityEdgeRecords, priorityEdgeDebug] = localGenerateEdgeCandidates( ...
            scaledGray, scaledEnhanced, grayImage, config, scale, true);
        [darkRecords, darkMask] = localGenerateDarkCandidates( ...
            scaledGray, grayImage, config, scale);
        [textRecords, textMask] = localGenerateTextClusterCandidates( ...
            scaledGray, grayImage, config, scale);

        candidateRecords = localAppendBranchCandidates( ...
            candidateRecords, edgeRecords, priorityEdgeRecords, darkRecords, textRecords);
        branchMasks = localUpdateBranchMasks( ...
            branchMasks, edgeDebug, priorityEdgeDebug, darkMask, textMask, scale);
    end

    candidateRecords = localSortAndPruneCandidates( ...
        candidateRecords, config.detection.maxCandidatesToKeep);
    metadata = localBuildMetadata(branchMasks, candidateRecords);

    if isempty(candidateRecords)
        bbox = [];
        return;
    end

    bbox = localPadBBox( ...
        candidateRecords(1).bbox, size(grayImage), config.detection.platePadding);
end

function branchMasks = localInitializeBranchMasks(imageSize)
    branchMasks = struct( ...
        "edgeMask", false(imageSize), ...
        "closedMask", false(imageSize), ...
        "openedMask", false(imageSize), ...
        "dilatedMask", false(imageSize), ...
        "plateMask", false(imageSize), ...
        "priorityEdgeMask", false(imageSize), ...
        "darkPlateMask", false(imageSize), ...
        "textClusterMask", false(imageSize), ...
        "multiScaleEdgeMask", false(imageSize));
end

function candidateRecords = localAppendBranchCandidates(candidateRecords, varargin)
    for i = 1:nargin - 1
        candidateRecords = [candidateRecords varargin{i}]; %#ok<AGROW>
    end
end

function branchMasks = localUpdateBranchMasks(branchMasks, edgeDebug, priorityEdgeDebug, darkMask, textMask, scale)
    if scale == 1
        branchMasks.edgeMask = edgeDebug.edgeMask;
        branchMasks.closedMask = edgeDebug.closedMask;
        branchMasks.openedMask = edgeDebug.openedMask;
        branchMasks.dilatedMask = edgeDebug.dilatedMask;
        branchMasks.plateMask = edgeDebug.plateMask;
        branchMasks.priorityEdgeMask = priorityEdgeDebug.plateMask;
    else
        branchMasks.multiScaleEdgeMask = branchMasks.multiScaleEdgeMask | ...
            edgeDebug.plateMask | priorityEdgeDebug.plateMask;
    end

    branchMasks.darkPlateMask = branchMasks.darkPlateMask | darkMask;
    branchMasks.textClusterMask = branchMasks.textClusterMask | textMask;
end

function metadata = localBuildMetadata(branchMasks, candidateRecords)
    if isempty(candidateRecords)
        topCandidates = candidateRecords;
        componentCount = 0;
        score = 0;
    else
        topCandidates = candidateRecords(1:min(5, numel(candidateRecords)));
        componentCount = numel(candidateRecords);
        score = candidateRecords(1).score;
    end

    metadata = struct( ...
        "edgeMask", branchMasks.edgeMask, ...
        "closedMask", branchMasks.closedMask, ...
        "openedMask", branchMasks.openedMask, ...
        "dilatedMask", branchMasks.dilatedMask, ...
        "plateMask", branchMasks.plateMask, ...
        "priorityEdgeMask", branchMasks.priorityEdgeMask, ...
        "darkPlateMask", branchMasks.darkPlateMask, ...
        "textClusterMask", branchMasks.textClusterMask, ...
        "multiScaleEdgeMask", branchMasks.multiScaleEdgeMask, ...
        "candidates", {candidateRecords}, ...
        "topCandidates", {topCandidates}, ...
        "componentCount", componentCount, ...
        "score", score);
end

function [candidateRecords, debug] = localGenerateEdgeCandidates( ...
        scaledGray, scaledEnhanced, originalGray, config, scale, usePriorityRoi)

    edgeMask = edge(scaledEnhanced, config.detection.edgeMethod);
    branchName = "edge_full";

    if usePriorityRoi
        edgeMask = edgeMask & localPriorityMask(size(scaledGray), config);
        branchName = "edge_priority";
    end

    closedMask = imclose(edgeMask, strel("rectangle", config.detection.closeKernel));
    openedMask = imopen(closedMask, strel("rectangle", config.detection.openKernel));
    dilatedMask = imdilate(openedMask, strel("rectangle", config.detection.dilateKernel));
    plateMask = imfill(dilatedMask, "holes");
    plateMask = bwareaopen(plateMask, config.detection.minCandidateAreaPixels);

    candidateRecords = localBuildCandidatesFromMask( ...
        plateMask, scaledGray, originalGray, config, scale, branchName);
    debug = struct( ...
        "edgeMask", localMapMaskToOriginal(edgeMask, size(originalGray)), ...
        "closedMask", localMapMaskToOriginal(closedMask, size(originalGray)), ...
        "openedMask", localMapMaskToOriginal(openedMask, size(originalGray)), ...
        "dilatedMask", localMapMaskToOriginal(dilatedMask, size(originalGray)), ...
        "plateMask", localMapMaskToOriginal(plateMask, size(originalGray)));
end

function [candidateRecords, mappedMask] = localGenerateDarkCandidates( ...
        scaledGray, originalGray, config, scale)

    roiBox = localPriorityBox(size(scaledGray), config);
    mask = false(size(scaledGray));
    roiImage = imcrop(scaledGray, roiBox);

    if ~isempty(roiImage)
        darkThreshold = prctile(double(roiImage(:)), 42);
        roiMask = roiImage <= darkThreshold;
        roiMask = imclose(roiMask, strel("rectangle", [5 17]));
        roiMask = imopen(roiMask, strel("rectangle", [3 5]));
        roiMask = imfill(roiMask, "holes");
        roiMask = bwareaopen(roiMask, config.detection.minCandidateAreaPixels);
        mask = localPasteRoiMask(mask, roiMask, roiBox);
    end

    candidateRecords = localBuildCandidatesFromMask( ...
        mask, scaledGray, originalGray, config, scale, "dark_priority");
    mappedMask = localMapMaskToOriginal(mask, size(originalGray));
end

function [candidateRecords, mappedMask] = localGenerateTextClusterCandidates( ...
        scaledGray, originalGray, config, scale)

    roiBox = localPriorityBox(size(scaledGray), config);
    mask = false(size(scaledGray));
    roiImage = imcrop(scaledGray, roiBox);

    if ~isempty(roiImage)
        equalizedRoi = adapthisteq(roiImage);
        roiMask = imbinarize( ...
            equalizedRoi, "adaptive", "ForegroundPolarity", "bright", "Sensitivity", 0.44);
        roiMask = imopen(roiMask, strel("rectangle", [2 2]));
        roiMask = imclose(roiMask, strel("rectangle", [5 21]));
        roiMask = imdilate(roiMask, strel("rectangle", [3 9]));
        roiMask = bwareaopen( ...
            roiMask, max(30, round(config.detection.minCandidateAreaPixels * 0.75)));
        mask = localPasteRoiMask(mask, roiMask, roiBox);
    end

    candidateRecords = localBuildCandidatesFromMask( ...
        mask, scaledGray, originalGray, config, scale, "text_priority");
    mappedMask = localMapMaskToOriginal(mask, size(originalGray));
end

function candidateRecords = localBuildCandidatesFromMask( ...
        mask, scaledGray, originalGray, config, scale, branchName)

    stats = regionprops( ...
        "table", mask, scaledGray, "BoundingBox", "Area", "Extent", "Solidity", "MeanIntensity");
    candidateRecords = localEmptyCandidateRecords();
    mappedMask = localMapMaskToOriginal(mask, size(originalGray));

    for i = 1:height(stats)
        if stats.Area(i) < config.detection.minCandidateAreaPixels
            continue;
        end

        mappedBox = localMapBoxToOriginal(stats.BoundingBox(i, :), scale, size(originalGray));
        mappedBox = localExpandCandidateBoxForBranch(mappedBox, branchName, size(originalGray));

        candidateMask = imcrop(mappedMask, mappedBox);
        candidateImage = imcrop(originalGray, mappedBox);
        if isempty(candidateMask) || isempty(candidateImage)
            continue;
        end

        [candidateRecord, isValid] = localScoreCandidate( ...
            mappedBox, candidateMask, candidateImage, size(originalGray), branchName, scale, config);
        if isValid
            candidateRecords(end+1) = candidateRecord; %#ok<AGROW>
        end
    end
end

function mappedBox = localExpandCandidateBoxForBranch(mappedBox, branchName, imageSize)
    switch string(branchName)
        case "text_priority"
            mappedBox = localExpandCandidateBox(mappedBox, [1.70 2.20], imageSize);
        case "dark_priority"
            mappedBox = localExpandCandidateBox(mappedBox, [1.25 1.35], imageSize);
    end
end

function [candidateRecord, isValid] = localScoreCandidate( ...
        bbox, candidateMask, candidateImage, originalImageSize, branchName, scale, config)

    fullImageHeight = originalImageSize(1);
    fullImageWidth = originalImageSize(2);

    areaRatio = (bbox(3) * bbox(4)) / max(fullImageHeight * fullImageWidth, eps);
    widthRatio = bbox(3) / max(fullImageWidth, eps);
    heightRatio = bbox(4) / max(fullImageHeight, eps);
    aspectRatio = bbox(3) / max(bbox(4), eps);
    verticalCenterRatio = (bbox(2) + bbox(4) / 2) / max(fullImageHeight, eps);
    horizontalCenterRatio = (bbox(1) + bbox(3) / 2) / max(fullImageWidth, eps);

    plateFeatures = extractPlateFeatures(candidateMask, candidateImage);
    rectangularity = plateFeatures.extent;
    solidity = plateFeatures.solidity;
    edgeDensity = plateFeatures.edgeDensity;
    contrastScore = min(plateFeatures.contrastScore, 1);
    characterTextureScore = plateFeatures.characterTextureScore;
    plateContrastScore = plateFeatures.plateContrastScore;
    alignmentScore = plateFeatures.componentAlignmentScore;
    textComponentCount = plateFeatures.textComponentCount;
    emptyRegionPenalty = plateFeatures.emptyRegionPenalty;

    isValid = areaRatio >= config.detection.minAreaRatio && ...
        areaRatio <= config.detection.maxAreaRatio && ...
        widthRatio >= config.detection.minWidthRatio && ...
        widthRatio <= config.detection.maxWidthRatio && ...
        heightRatio >= config.detection.minHeightRatio && ...
        heightRatio <= config.detection.maxHeightRatio && ...
        verticalCenterRatio >= config.detection.minVerticalCenterRatio && ...
        verticalCenterRatio <= config.detection.maxVerticalCenterRatio && ...
        horizontalCenterRatio >= config.detection.minHorizontalCenterRatio && ...
        horizontalCenterRatio <= config.detection.maxHorizontalCenterRatio && ...
        rectangularity >= config.detection.minExtent && ...
        solidity >= config.detection.minSolidity;

    candidateRecord = [];
    if ~isValid
        return;
    end

    bestScore = -inf;
    bestProfile = "";
    bestBreakdown = struct();

    for profile = config.detection.profiles
        aspectFit = localProfileFit( ...
            aspectRatio, profile.targetAspectRatio, profile.minAspectRatio, profile.maxAspectRatio);
        widthFit = localProfileFit( ...
            widthRatio, profile.targetWidthRatio, profile.minWidthRatio, profile.maxWidthRatio);
        heightFit = localProfileFit( ...
            heightRatio, profile.targetHeightRatio, profile.minHeightRatio, profile.maxHeightRatio);
        areaFit = localProfileFit( ...
            areaRatio, profile.targetAreaRatio, profile.minAreaRatio, profile.maxAreaRatio);

        geometryScore = mean([aspectFit widthFit heightFit areaFit]);
        shapeScore = mean([ ...
            localThresholdScore(rectangularity, 0.22, 0.78) ...
            localThresholdScore(solidity, 0.18, 0.85)]);
        textureScore = mean([ ...
            localThresholdScore(edgeDensity, 0.04, 0.32) ...
            contrastScore ...
            characterTextureScore]);
        locationScore = mean([ ...
            localProfileFit(verticalCenterRatio, config.detection.targetVerticalCenterRatio, 0.42, 0.92) ...
            localProfileFit(horizontalCenterRatio, config.detection.targetHorizontalCenterRatio, 0.12, 0.88)]);
        vehiclePositionScore = mean([ ...
            localProfileFit(verticalCenterRatio, 0.78, 0.48, 0.92) ...
            localProfileFit(horizontalCenterRatio, 0.50, 0.22, 0.78)]);
        evidenceScore = mean([ ...
            characterTextureScore ...
            plateContrastScore ...
            alignmentScore ...
            vehiclePositionScore]);
        branchScore = localBranchScore(branchName, scale);
        finalScore = 0.24 * geometryScore + 0.10 * shapeScore + 0.14 * textureScore + ...
            0.10 * locationScore + 0.08 * branchScore + 0.26 * evidenceScore - ...
            0.08 * emptyRegionPenalty;
        finalScore = max(0, min(1, finalScore));

        if finalScore > bestScore
            bestScore = finalScore;
            bestProfile = string(profile.name);
            bestBreakdown = struct( ...
                "geometry", geometryScore, ...
                "shape", shapeScore, ...
                "texture", textureScore, ...
                "location", locationScore, ...
                "vehiclePosition", vehiclePositionScore, ...
                "characterTexture", characterTextureScore, ...
                "plateContrast", plateContrastScore, ...
                "componentAlignment", alignmentScore, ...
                "emptyRegionPenalty", emptyRegionPenalty, ...
                "branch", branchScore, ...
                "evidence", evidenceScore, ...
                "final", finalScore);
        end
    end

    candidateRecord = struct( ...
        "bbox", bbox, ...
        "score", bestScore, ...
        "branchName", string(branchName), ...
        "profileName", bestProfile, ...
        "scale", scale, ...
        "areaRatio", areaRatio, ...
        "widthRatio", widthRatio, ...
        "heightRatio", heightRatio, ...
        "aspectRatio", aspectRatio, ...
        "edgeDensity", edgeDensity, ...
        "contrastScore", contrastScore, ...
        "rectangularity", rectangularity, ...
        "solidity", solidity, ...
        "characterTextureScore", characterTextureScore, ...
        "plateContrastScore", plateContrastScore, ...
        "componentAlignmentScore", alignmentScore, ...
        "vehiclePositionScore", bestBreakdown.vehiclePosition, ...
        "textComponentCount", textComponentCount, ...
        "emptyRegionPenalty", emptyRegionPenalty, ...
        "verticalCenterRatio", verticalCenterRatio, ...
        "horizontalCenterRatio", horizontalCenterRatio, ...
        "scoreBreakdown", bestBreakdown);
end

function score = localBranchScore(branchName, scale)
    switch string(branchName)
        case "edge_priority"
            base = 1.00;
        case "text_priority"
            base = 0.96;
        case "dark_priority"
            base = 0.93;
        otherwise
            base = 0.86;
    end

    scaleOffset = abs(scale - 1.0);
    score = max(0.70, base - 0.10 * scaleOffset);
end

function score = localProfileFit(value, target, minValue, maxValue)
    insideRange = value >= minValue && value <= maxValue;
    span = max(maxValue - minValue, eps);
    centerPenalty = abs(value - target) / span;

    if insideRange
        score = max(0.35, 1 - 0.90 * centerPenalty);
        return;
    end

    if value < minValue
        overflow = (minValue - value) / max(abs(minValue), eps);
    else
        overflow = (value - maxValue) / max(abs(maxValue), eps);
    end
    score = max(0, 0.30 - overflow);
end

function score = localThresholdScore(value, lowTarget, highTarget)
    if value <= lowTarget
        score = max(0, value / max(lowTarget, eps));
    elseif value >= highTarget
        score = max(0, 1 - (value - highTarget) / max(1 - highTarget, eps));
    else
        score = 1;
    end
end

function candidateRecords = localSortAndPruneCandidates(candidateRecords, maxCandidatesToKeep)
    if isempty(candidateRecords)
        return;
    end

    [~, sortIdx] = sort([candidateRecords.score], "descend");
    candidateRecords = candidateRecords(sortIdx);

    keepMask = false(1, numel(candidateRecords));
    keptCount = 0;

    for i = 1:numel(candidateRecords)
        overlapsExisting = false;
        for j = find(keepMask)
            if localIoU(candidateRecords(i).bbox, candidateRecords(j).bbox) >= 0.60
                overlapsExisting = true;
                break;
            end
        end

        if overlapsExisting
            continue;
        end

        keepMask(i) = true;
        keptCount = keptCount + 1;
        if keptCount >= maxCandidatesToKeep
            break;
        end
    end

    candidateRecords = candidateRecords(keepMask);
end

function mask = localPriorityMask(imageSize, config)
    roiBox = localPriorityBox(imageSize, config);
    mask = false(imageSize);
    mask = localPasteRoiMask(mask, true(roiBox(4) + 1, roiBox(3) + 1), roiBox);
end

function roiBox = localPriorityBox(imageSize, config)
    normalizedRoi = config.detection.priorityRoi;
    imageHeight = imageSize(1);
    imageWidth = imageSize(2);

    x1 = max(1, floor(normalizedRoi(1) * imageWidth));
    y1 = max(1, floor(normalizedRoi(2) * imageHeight));
    x2 = min(imageWidth, ceil(normalizedRoi(3) * imageWidth));
    y2 = min(imageHeight, ceil(normalizedRoi(4) * imageHeight));
    roiBox = [x1 y1 max(1, x2 - x1) max(1, y2 - y1)];
end

function pastedMask = localPasteRoiMask(mask, roiMask, roiBox)
    pastedMask = mask;
    yIdx = roiBox(2):(roiBox(2) + size(roiMask, 1) - 1);
    xIdx = roiBox(1):(roiBox(1) + size(roiMask, 2) - 1);
    yIdx = yIdx(yIdx >= 1 & yIdx <= size(mask, 1));
    xIdx = xIdx(xIdx >= 1 & xIdx <= size(mask, 2));
    pastedMask(yIdx, xIdx) = roiMask(1:numel(yIdx), 1:numel(xIdx));
end

function mappedMask = localMapMaskToOriginal(mask, originalSize)
    mappedMask = imresize(logical(mask), originalSize, "nearest");
end

function mappedBox = localMapBoxToOriginal(box, scale, imageSize)
    if scale == 0
        mappedBox = [];
        return;
    end

    mappedBox = [box(1) box(2) box(3) box(4)] / scale;
    mappedBox(1) = max(1, floor(mappedBox(1)));
    mappedBox(2) = max(1, floor(mappedBox(2)));
    mappedBox(3) = max(1, ceil(mappedBox(3)));
    mappedBox(4) = max(1, ceil(mappedBox(4)));

    if mappedBox(1) + mappedBox(3) > imageSize(2)
        mappedBox(3) = max(1, imageSize(2) - mappedBox(1));
    end
    if mappedBox(2) + mappedBox(4) > imageSize(1)
        mappedBox(4) = max(1, imageSize(1) - mappedBox(2));
    end
end

function expandedBox = localExpandCandidateBox(box, scaleFactors, imageSize)
    centerX = box(1) + box(3) / 2;
    centerY = box(2) + box(4) / 2;
    expandedWidth = box(3) * scaleFactors(1);
    expandedHeight = box(4) * scaleFactors(2);

    x1 = max(1, floor(centerX - expandedWidth / 2));
    y1 = max(1, floor(centerY - expandedHeight / 2));
    x2 = min(imageSize(2), ceil(centerX + expandedWidth / 2));
    y2 = min(imageSize(1), ceil(centerY + expandedHeight / 2));
    expandedBox = [x1 y1 max(1, x2 - x1) max(1, y2 - y1)];
end

function value = localIoU(boxA, boxB)
    x1 = max(boxA(1), boxB(1));
    y1 = max(boxA(2), boxB(2));
    x2 = min(boxA(1) + boxA(3), boxB(1) + boxB(3));
    y2 = min(boxA(2) + boxA(4), boxB(2) + boxB(4));

    intersectionArea = max(0, x2 - x1) * max(0, y2 - y1);
    areaA = boxA(3) * boxA(4);
    areaB = boxB(3) * boxB(4);
    unionArea = areaA + areaB - intersectionArea;

    if unionArea == 0
        value = 0;
    else
        value = intersectionArea / unionArea;
    end
end

function records = localEmptyCandidateRecords()
    records = struct( ...
        "bbox", {}, ...
        "score", {}, ...
        "branchName", {}, ...
        "profileName", {}, ...
        "scale", {}, ...
        "areaRatio", {}, ...
        "widthRatio", {}, ...
        "heightRatio", {}, ...
        "aspectRatio", {}, ...
        "edgeDensity", {}, ...
        "contrastScore", {}, ...
        "rectangularity", {}, ...
        "solidity", {}, ...
        "characterTextureScore", {}, ...
        "plateContrastScore", {}, ...
        "componentAlignmentScore", {}, ...
        "vehiclePositionScore", {}, ...
        "textComponentCount", {}, ...
        "emptyRegionPenalty", {}, ...
        "verticalCenterRatio", {}, ...
        "horizontalCenterRatio", {}, ...
        "scoreBreakdown", {});
end

function bbox = localPadBBox(bbox, imageSize, paddingRatio)
    paddingX = bbox(3) * paddingRatio;
    paddingY = bbox(4) * paddingRatio;

    x1 = max(1, floor(bbox(1) - paddingX));
    y1 = max(1, floor(bbox(2) - paddingY));
    x2 = min(imageSize(2), ceil(bbox(1) + bbox(3) + paddingX));
    y2 = min(imageSize(1), ceil(bbox(2) + bbox(4) + paddingY));

    bbox = [x1 y1 max(1, x2 - x1) max(1, y2 - y1)];
end
