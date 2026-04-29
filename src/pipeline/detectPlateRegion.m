function [bbox, metadata] = detectPlateRegion(grayImage, config)
    % DETECTPLATEREGION Main entry point for detecting a license plate.
    % It enhances the image, scans across multiple scales, and runs several 
    % detection strategies (edge, dark regions, text clusters) to find the 
    % highest-scoring bounding box that looks like a license plate.

    config = validateConfig(config);
    grayImage = im2uint8(grayImage);
    enhancedImage = enhanceContrast( ...
        reduceNoise(normalizeLighting(grayImage, config), config), ...
        config);
    imageSize = size(grayImage);

    candidateRecords = localEmptyCandidateRecords();
    branchMasks = localInitializeBranchMasks(imageSize);

    % Loop through different image scales to make detection scale-invariant
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
        [mserRecords, mserMask] = localGenerateMserCandidates( ...
            scaledGray, grayImage, config, scale);

        candidateRecords = localAppendBranchCandidates( ...
            candidateRecords, edgeRecords, priorityEdgeRecords, darkRecords, textRecords, mserRecords);
        branchMasks = localUpdateBranchMasks( ...
            branchMasks, edgeDebug, priorityEdgeDebug, darkMask, textMask, mserMask, scale);
    end

    candidateRecords = localSortAndPruneCandidates( ...
        candidateRecords, config.detection.maxCandidatesToKeep);
    metadata = localBuildMetadata(branchMasks, candidateRecords);
    localShowDetectionCandidateDebug(grayImage, candidateRecords, config);

    if isempty(candidateRecords)
        bbox = [];
        return;
    end

    bbox = localPadBBox( ...
        candidateRecords(1).bbox, size(grayImage), config.detection.platePadding);
end

function branchMasks = localInitializeBranchMasks(imageSize)
    % LOCALINITIALIZEBRANCHMASKS Creates empty logical matrices to keep track 
    % of where potential plate regions were found during processing.
    
    branchMasks = struct( ...
        "edgeMask", false(imageSize), ...
        "closedMask", false(imageSize), ...
        "openedMask", false(imageSize), ...
        "dilatedMask", false(imageSize), ...
        "plateMask", false(imageSize), ...
        "priorityEdgeMask", false(imageSize), ...
        "darkPlateMask", false(imageSize), ...
        "textClusterMask", false(imageSize), ...
        "mserMask", false(imageSize), ...
        "multiScaleEdgeMask", false(imageSize));
end

function candidateRecords = localAppendBranchCandidates(candidateRecords, varargin)
    % LOCALAPPENDBRANCHCANDIDATES Utility function to concatenate multiple arrays 
    % of candidate structures into a single array.

    for i = 1:nargin - 1
        candidateRecords = [candidateRecords varargin{i}]; %#ok<AGROW>
    end
end

function branchMasks = localUpdateBranchMasks(branchMasks, edgeDebug, priorityEdgeDebug, darkMask, textMask, mserMask, scale)
    % LOCALUPDATEBRANCHMASKS Updates the debugging masks with results from 
    % the current scale iteration.

    if scale == 1
        branchMasks.edgeMask = edgeDebug.edgeMask;
        branchMasks.closedMask = edgeDebug.closedMask;
        branchMasks.openedMask = edgeDebug.openedMask;
        branchMasks.dilatedMask = edgeDebug.dilatedMask;
        branchMasks.plateMask = edgeDebug.plateMask;
        branchMasks.priorityEdgeMask = priorityEdgeDebug.plateMask;
        branchMasks.mserMask = branchMasks.mserMask | mserMask;
    else
        branchMasks.multiScaleEdgeMask = branchMasks.multiScaleEdgeMask | ...
            edgeDebug.plateMask | priorityEdgeDebug.plateMask;
    end

    branchMasks.darkPlateMask = branchMasks.darkPlateMask | darkMask;
    branchMasks.textClusterMask = branchMasks.textClusterMask | textMask;
end

function metadata = localBuildMetadata(branchMasks, candidateRecords)
    % LOCALBUILDMETADATA Packages the masks, candidates, and top scores 
    % into a struct for debugging, visualization, or external logging.

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
        "mserMask", branchMasks.mserMask, ...
        "multiScaleEdgeMask", branchMasks.multiScaleEdgeMask, ...
        "candidates", {candidateRecords}, ...
        "topCandidates", {topCandidates}, ...
        "componentCount", componentCount, ...
        "score", score);
end

function [candidateRecords, debug] = localGenerateEdgeCandidates( ...
        scaledGray, scaledEnhanced, originalGray, config, scale, usePriorityRoi)

    % LOCALGENERATEEDGECANDIDATES Finds license plate candidates by looking 
    % for regions with high edge density (which happens where text is).
    % Uses morphological operations to cluster edges into solid rectangles.

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
    % LOCALGENERATEDARKCANDIDATES Looks for license plates that appear as 
    % distinct dark rectangles against a lighter car body.

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
    % LOCALGENERATETEXTCLUSTERCANDIDATES Targets high-contrast text regions 
    % by aggressively applying adaptive thresholding to find characters.

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

function [candidateRecords, mappedMask] = localGenerateMserCandidates( ...
        scaledGray, originalGray, config, scale)

    candidateRecords = localEmptyCandidateRecords();
    mappedMask = false(size(originalGray));

    if ~isfield(config.detection, "mserEnabled") || ~config.detection.mserEnabled
        return;
    end
    if ~exist("detectMSERFeatures", "file")
        return;
    end

    mserRegions = detectMSERFeatures(scaledGray, ...
        "RegionAreaRange", [config.detection.mserMinAreaPixels config.detection.mserMaxAreaPixels]);
    if mserRegions.Count == 0
        return;
    end

    markerMask = false(size(scaledGray));
    regionLocations = round(mserRegions.Location);
    regionLocations(:, 1) = max(1, min(size(markerMask, 2), regionLocations(:, 1)));
    regionLocations(:, 2) = max(1, min(size(markerMask, 1), regionLocations(:, 2)));
    linearIdx = sub2ind(size(markerMask), regionLocations(:, 2), regionLocations(:, 1));
    markerMask(linearIdx) = true;

    markerMask = imdilate(markerMask, strel("rectangle", [3 3]));
    mserMask = imclose(markerMask, strel("rectangle", [5 17]));
    mserMask = imdilate(mserMask, strel("rectangle", [3 11]));
    mserMask = bwareaopen(mserMask, config.detection.minCandidateAreaPixels);
    mserMask = localFilterMserMaskByAspect(mserMask, config);

    candidateRecords = localBuildCandidatesFromMask( ...
        mserMask, scaledGray, originalGray, config, scale, "mser_text");
    mappedMask = localMapMaskToOriginal(mserMask, size(originalGray));
end

function filteredMask = localFilterMserMaskByAspect(mask, config)
    % LOCALFILTERMSERMASKBYASPECT Filters MSER regions based on their aspect ratio.

    filteredMask = false(size(mask));
    stats = regionprops("table", mask, "BoundingBox", "Area");
    for i = 1:height(stats)
        box = stats.BoundingBox(i, :);
        aspectRatio = box(3) / max(box(4), eps);
        if aspectRatio >= config.detection.mserMinAspectRatio && ...
                aspectRatio <= config.detection.mserMaxAspectRatio
            componentMask = false(size(mask));
            componentMask = localFillBox(componentMask, box);
            filteredMask = filteredMask | componentMask;
        end
    end
end

function mask = localFillBox(mask, box)
    % LOCALFILLBOX Sets the pixels within a bounding box to true in a logical mask.

    x1 = max(1, floor(box(1)));
    y1 = max(1, floor(box(2)));
    x2 = min(size(mask, 2), ceil(box(1) + box(3) - 1));
    y2 = min(size(mask, 1), ceil(box(2) + box(4) - 1));
    mask(y1:y2, x1:x2) = true;
end

function candidateRecords = localBuildCandidatesFromMask( ...
        mask, scaledGray, originalGray, config, scale, branchName)
    % LOCALBUILDCANDIDATESFROMMASK Extracts bounding boxes from binary masks, 
    % translates them to the original image coordinates, and grades them.

    stats = regionprops( ...
        "table", mask, scaledGray, "BoundingBox", "Area", "Extent", "Solidity", "MeanIntensity");
    candidateRecords = localEmptyCandidateRecords();
    mappedMask = localMapMaskToOriginal(mask, size(originalGray));

    for i = 1:height(stats)
        if stats.Area(i) < config.detection.minCandidateAreaPixels
            continue;
        end

        mappedBox = localMapBoxToOriginal(stats.BoundingBox(i, :), scale, size(originalGray));
        mappedBox = localExpandCandidateBoxForBranch(mappedBox, branchName, size(originalGray), config);

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

function mappedBox = localExpandCandidateBoxForBranch(mappedBox, branchName, imageSize, config)
    % LOCALEXPANDCANDIDATEBOXFORBRANCH Certain detection algorithms typically 
    % underestimate the plate size, so this manually inflates the box.

    switch string(branchName)
        case "text_priority"
            mappedBox = localExpandCandidateBox(mappedBox, [1.70 2.20], imageSize);
        case "dark_priority"
            mappedBox = localExpandCandidateBox(mappedBox, [1.25 1.35], imageSize);
        case "mser_text"
            mappedBox = localExpandCandidateBox(mappedBox, config.detection.mserExpand, imageSize); 
    end
end

function [candidateRecord, isValid] = localScoreCandidate( ...
    bbox, candidateMask, candidateImage, originalImageSize, branchName, scale, config)
    % LOCALSCORECANDIDATE The core evaluation function. It checks if the candidate 
    % meets strict geometric constraints (aspect ratio, width, height) and then 
    % scores it based on texture, contrast, and shape.

    fullImageHeight = originalImageSize(1);
    fullImageWidth = originalImageSize(2);

    areaRatio = (bbox(3) * bbox(4)) / max(fullImageHeight * fullImageWidth, eps);
    widthRatio = bbox(3) / max(fullImageWidth, eps);
    heightRatio = bbox(4) / max(fullImageHeight, eps);
    aspectRatio = bbox(3) / max(bbox(4), eps);
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
        rectangularity >= config.detection.minExtent && ...
        solidity >= config.detection.minSolidity;

    candidateRecord = [];
    if ~isValid
        return;
    end

    bestScore = -inf;
    bestProfile = "";
    bestBreakdown = struct();

    % Test against different standard plate profiles (e.g., standard vs squarish)
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
        evidenceScore = mean([ ...
            characterTextureScore ...
            plateContrastScore ...
            alignmentScore]);
        branchScore = localBranchScore(branchName, scale);

        countScore = localTextComponentCountScore(textComponentCount);
        coverageScore = localTextCoverageScore(candidateMask);
        splitPenalty = localSplitFragmentPenalty(candidateMask);

        % Final weighted score calculation
        finalScore = 0.21 * geometryScore + 0.09 * shapeScore + 0.12 * textureScore + ...
            0.10 * branchScore + 0.22 * evidenceScore + ...
            0.12 * countScore + 0.15 * coverageScore - ...
            0.06 * emptyRegionPenalty - 0.06 * splitPenalty;

        if finalScore > bestScore
            bestScore = finalScore;
            bestProfile = string(profile.name);
            bestBreakdown = struct( ...
                "geometry", geometryScore, ...
                "shape", shapeScore, ...
                "texture", textureScore, ...
                "characterTexture", characterTextureScore, ...
                "plateContrast", plateContrastScore, ...
                "componentAlignment", alignmentScore, ...
                "textCount", textComponentCount, ...
                "countScore", countScore, ...
                "coverageScore", coverageScore, ...
                "splitPenalty", splitPenalty, ...
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
        "textComponentCount", textComponentCount, ...
        "emptyRegionPenalty", emptyRegionPenalty, ...
        "scoreBreakdown", bestBreakdown);
end

function score = localBranchScore(branchName, scale)
    % LOCALBRANCHSCORE Assigns a small bias/confidence score depending on 
    % which detection method generated this candidate. Edge priority is highest.
    switch string(branchName)
        case "edge_priority"
            base = 1.00;
        case "text_priority"
            base = 0.96;
        case "dark_priority"
            base = 0.93;
        case "mser_text"
            base = 0.98;
        otherwise
            base = 0.86;
    end

    scaleOffset = abs(scale - 1.0);
    score = max(0.70, base - 0.10 * scaleOffset);
end

function score = localProfileFit(value, target, minValue, maxValue)
    % LOCALPROFILEFIT Calculates how close a value is to a target profile value, 
    % returning a high score for close matches and penalizing outliers.

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

function score = localTextComponentCountScore(textComponentCount)
    if textComponentCount >= 5 && textComponentCount <= 8
        score = 1.0;
    elseif textComponentCount >= 3 && textComponentCount <= 10
        distance = min(abs(textComponentCount - 5), abs(textComponentCount - 8));
        score = max(0.35, 0.90 - 0.15 * distance);
    else
        score = 0.10;
    end
end

function score = localTextCoverageScore(candidateMask)
    mask = logical(candidateMask);
    if ~any(mask(:))
        score = 0;
        return;
    end

    profileX = sum(mask, 1);
    activeCols = find(profileX > 0);
    if isempty(activeCols)
        score = 0;
        return;
    end

    coverageRatio = (activeCols(end) - activeCols(1) + 1) / max(size(mask, 2), 1);
    score = max(0, min(1, (coverageRatio - 0.25) / 0.55));
end

function penalty = localSplitFragmentPenalty(candidateMask)
    mask = logical(candidateMask);
    cc = bwconncomp(mask);
    if cc.NumObjects <= 1
        penalty = 0;
        return;
    end

    stats = regionprops(cc, "Area");
    areas = sort([stats.Area], "descend");
    totalArea = sum(areas);
    if totalArea <= 0
        penalty = 0;
        return;
    end

    dominantShare = areas(1) / totalArea;
    if numel(areas) >= 2
        secondShare = areas(2) / totalArea;
    else
        secondShare = 0;
    end

    penalty = max(0, min(1, 0.7 * (1 - dominantShare) + 0.3 * secondShare));
end

function candidateRecords = localSortAndPruneCandidates(candidateRecords, maxCandidatesToKeep)
    % LOCALSORTANDPRUNECANDIDATES Performs Non-Maximum Suppression (NMS).
    % Sorts candidates by score, then removes any lower-scoring candidates 
    % that heavily overlap with higher-scoring ones.

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
    % LOCALPRIORITYMASK Generates a full-image-sized binary mask where only 
    % the priority ROI (likely area for a plate) is set to true.

    roiBox = localPriorityBox(imageSize, config);
    mask = false(imageSize);
    mask = localPasteRoiMask(mask, true(roiBox(4) + 1, roiBox(3) + 1), roiBox);
end

function roiBox = localPriorityBox(imageSize, config)
    % LOCALPRIORITYBOX Converts normalized ROI coordinates from the config 
    % into absolute pixel coordinates based on the image size.

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
    % LOCALPASTEROIMASK Inserts a smaller mask matrix into a larger 
    % full-image mask at a specified bounding box location.

    pastedMask = mask;
    yIdx = roiBox(2):(roiBox(2) + size(roiMask, 1) - 1);
    xIdx = roiBox(1):(roiBox(1) + size(roiMask, 2) - 1);
    yIdx = yIdx(yIdx >= 1 & yIdx <= size(mask, 1));
    xIdx = xIdx(xIdx >= 1 & xIdx <= size(mask, 2));
    pastedMask(yIdx, xIdx) = roiMask(1:numel(yIdx), 1:numel(xIdx));
end

function mappedMask = localMapMaskToOriginal(mask, originalSize)
    % LOCALMAPMASKTOORIGINAL Resizes a candidate mask generated at a different 
    % scale back to the original image dimensions.

    mappedMask = imresize(logical(mask), originalSize, "nearest");
end

function mappedBox = localMapBoxToOriginal(box, scale, imageSize)
    % LOCALMAPBOXTOORIGINAL Scales bounding box coordinates from a scaled 
    % image back to match the original image size.

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
    % LOCALEXPANDCANDIDATEBOX Grows a bounding box outwards from its center 
    % based on provided width and height multipliers.

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
    % LOCALIOU Calculates "Intersection over Union". This returns a percentage 
    % of how much two bounding boxes overlap, used to filter out duplicates.

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
        "textComponentCount", {}, ...
        "emptyRegionPenalty", {}, ...
        "scoreBreakdown", {});
end

function bbox = localPadBBox(bbox, imageSize, paddingRatio)
    % LOCALPADBBOX Adds a final proportional margin around the best bounding box
    % to ensure the edges of the plate arent cut off before OCR

    paddingX = bbox(3) * paddingRatio;
    paddingY = bbox(4) * paddingRatio;

    x1 = max(1, floor(bbox(1) - paddingX));
    y1 = max(1, floor(bbox(2) - paddingY));
    x2 = min(imageSize(2), ceil(bbox(1) + bbox(3) + paddingX));
    y2 = min(imageSize(1), ceil(bbox(2) + bbox(4) + paddingY));

    bbox = [x1 y1 max(1, x2 - x1) max(1, y2 - y1)];
end

function localShowDetectionCandidateDebug(grayImage, candidateRecords, config)
    % LOCALSHOWDETECTIONCANDIDATEDEBUG If debugging is enabled, this function will print a summary of the
    % candidates and show a figure with the top candidates overlaid on the original image.

    if isempty(candidateRecords) || ~isfield(config, "debug")
        return;
    end

    if isfield(config.debug, "printDetectionCandidateSummary") && ...
            config.debug.printDetectionCandidateSummary
        showDetectionCandidateSummary(candidateRecords, config);
    end

    if isfield(config.debug, "showDetectionCandidatesFigure") && ...
            config.debug.showDetectionCandidatesFigure
        showDetectionCandidateDebugFigure(grayImage, candidateRecords, config);
    end
end
