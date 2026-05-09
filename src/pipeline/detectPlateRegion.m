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
            scaledGray, grayImage, config, scale, "bright", "text_priority");
        [darkTextRecords, darkTextMask] = localGenerateTextClusterCandidates( ...
            scaledGray, grayImage, config, scale, "dark", "text_dark_priority");
        [mserRecords, mserMask] = localGenerateMserCandidates( ...
            scaledGray, grayImage, config, scale);

        candidateRecords = localAppendBranchCandidates( ...
            candidateRecords, edgeRecords, priorityEdgeRecords, darkRecords, textRecords, darkTextRecords, mserRecords);
        branchMasks = localUpdateBranchMasks( ...
            branchMasks, edgeDebug, priorityEdgeDebug, darkMask, textMask | darkTextMask, mserMask, scale);
    end

    candidateRecords = localSortAndPruneCandidates( ...
        candidateRecords, config.detection.maxCandidatesToKeep);
    metadata = localBuildMetadata(branchMasks, candidateRecords, config);
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

function metadata = localBuildMetadata(branchMasks, candidateRecords, config)
    % LOCALBUILDMETADATA Packages the masks, candidates, and top scores 
    % into a struct for debugging, visualization, or external logging.

    if isempty(candidateRecords)
        topCandidates = candidateRecords;
        componentCount = 0;
        score = 0;
    else
        topCount = max(1, round(double(config.reranking.maxCandidatesToEvaluate)));
        topCandidates = candidateRecords(1:min(topCount, numel(candidateRecords)));
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
    plateMask = bwareaopen(plateMask, localMaskAreaThreshold(size(plateMask), config, 1.00));

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
        roiMask = bwareaopen(roiMask, localMaskAreaThreshold(size(roiMask), config, 1.00));
        mask = localPasteRoiMask(mask, roiMask, roiBox);
    end

    candidateRecords = localBuildCandidatesFromMask( ...
        mask, scaledGray, originalGray, config, scale, "dark_priority");
    mappedMask = localMapMaskToOriginal(mask, size(originalGray));
end

function [candidateRecords, mappedMask] = localGenerateTextClusterCandidates( ...
        scaledGray, originalGray, config, scale, polarity, branchName)
    % LOCALGENERATETEXTCLUSTERCANDIDATES Targets high-contrast text regions 
    % by aggressively applying adaptive thresholding to find characters.

    if nargin < 5 || strlength(string(polarity)) == 0
        polarity = "bright";
    end
    if nargin < 6 || strlength(string(branchName)) == 0
        branchName = "text_priority";
    end

    roiBox = localPriorityBox(size(scaledGray), config);
    mask = false(size(scaledGray));
    roiImage = imcrop(scaledGray, roiBox);

    if ~isempty(roiImage)
        equalizedRoi = adapthisteq(roiImage);
        roiMask = imbinarize( ...
            equalizedRoi, "adaptive", "ForegroundPolarity", polarity, "Sensitivity", 0.44);
        roiMask = imopen(roiMask, strel("rectangle", config.detection.textClusterOpenKernel));
        roiMask = imclose(roiMask, strel("rectangle", config.detection.textClusterCloseKernel));
        roiMask = imdilate(roiMask, strel("rectangle", config.detection.textClusterDilateKernel));
        roiMask = bwareaopen(roiMask, localMaskAreaThreshold(size(roiMask), config, 0.65));
        mask = localPasteRoiMask(mask, roiMask, roiBox);
    end

    candidateRecords = localBuildCandidatesFromMask( ...
        mask, scaledGray, originalGray, config, scale, branchName);
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

    try
        mserRegions = detectMSERFeatures(scaledGray, ...
            "RegionAreaRange", [config.detection.mserMinAreaPixels config.detection.mserMaxAreaPixels]);
    catch
        return;
    end
    if mserRegions.Count == 0
        return;
    end

    markerMask = false(size(scaledGray));
    regionLocations = round(mserRegions.Location);
    regionLocations(:, 1) = max(1, min(size(markerMask, 2), regionLocations(:, 1)));
    regionLocations(:, 2) = max(1, min(size(markerMask, 1), regionLocations(:, 2)));
    linearIdx = sub2ind(size(markerMask), regionLocations(:, 2), regionLocations(:, 1));
    markerMask(linearIdx) = true;

    markerMask = imdilate(markerMask, strel("rectangle", config.detection.mserSeedDilateKernel));
    mserMask = imclose(markerMask, strel("rectangle", config.detection.mserCloseKernel));
    mserMask = imdilate(mserMask, strel("rectangle", config.detection.mserDilateKernel));
    mserMask = bwareaopen(mserMask, localMaskAreaThreshold(size(mserMask), config, 0.75));
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
        case "edge_full"
            mappedBox = localExpandCandidateBox(mappedBox, config.detection.edgeFullExpand, imageSize);
        case "edge_priority"
            mappedBox = localExpandCandidateBox(mappedBox, config.detection.edgePriorityExpand, imageSize);
        case "text_priority"
            mappedBox = localExpandCandidateBox(mappedBox, [1.70 2.20], imageSize);
        case "text_dark_priority"
            mappedBox = localExpandCandidateBox(mappedBox, config.detection.textDarkExpand, imageSize);
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
    horizontalCenterRatio = (bbox(1) + bbox(3) / 2) / max(fullImageWidth, eps);
    verticalCenterRatio = (bbox(2) + bbox(4) / 2) / max(fullImageHeight, eps);
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
    layoutHint = string(plateFeatures.layoutHint);
    rowCountEstimate = double(plateFeatures.rowCountEstimate);
    singleLineAlignmentScore = localSafeFeatureValue(plateFeatures, "singleLineAlignmentScore");
    twoRowAlignmentScore = localSafeFeatureValue(plateFeatures, "twoRowAlignmentScore");

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
        profileName = string(profile.name);
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
        if profileName == "two_row"
            profileAlignmentScore = max(twoRowAlignmentScore, 0.70 * alignmentScore);
        else
            profileAlignmentScore = max(singleLineAlignmentScore, 0.70 * alignmentScore);
        end
        textureScore = mean([ ...
            localThresholdScore(edgeDensity, 0.04, 0.32) ...
            contrastScore ...
            characterTextureScore]);
        evidenceScore = mean([ ...
            characterTextureScore ...
            plateContrastScore ...
            profileAlignmentScore]);
        branchScore = localBranchScore(branchName, scale);

        countScore = localTextComponentCountScore(textComponentCount, profileName);
        coverageScore = localTextCoverageScore(candidateMask);
        splitPenalty = localSplitFragmentPenalty(candidateMask, layoutHint, profileName);
        positionScore = localCandidatePositionScore(horizontalCenterRatio, verticalCenterRatio, config);
        spanScore = localCandidateSpanScore(widthRatio, heightRatio, aspectRatio, profile);
        [textBandScore, edgeClipPenalty] = localTextBandCompletenessScore(candidateImage, profileName);
        layoutScore = localLayoutMatchScore(layoutHint, rowCountEstimate, profileName);
        boundaryPenalty = localBoundaryPenalty(bbox, originalImageSize);

        % Final weighted score calculation
        finalScore = 0.16 * geometryScore + 0.07 * shapeScore + 0.10 * textureScore + ...
            0.08 * branchScore + 0.18 * evidenceScore + ...
            0.10 * countScore + 0.08 * coverageScore + 0.07 * positionScore + ...
            0.08 * spanScore + 0.10 * textBandScore + 0.06 * layoutScore - ...
            0.04 * emptyRegionPenalty - 0.04 * splitPenalty - 0.08 * edgeClipPenalty - ...
            0.14 * boundaryPenalty;

        if finalScore > bestScore
            bestScore = finalScore;
            bestProfile = profileName;
            bestBreakdown = struct( ...
                "geometry", geometryScore, ...
                "shape", shapeScore, ...
                "texture", textureScore, ...
                "characterTexture", characterTextureScore, ...
                "plateContrast", plateContrastScore, ...
                "componentAlignment", profileAlignmentScore, ...
                "textCount", textComponentCount, ...
                "countScore", countScore, ...
                "coverageScore", coverageScore, ...
                "positionScore", positionScore, ...
                "spanScore", spanScore, ...
                "layoutScore", layoutScore, ...
                "layoutHint", layoutHint, ...
                "rowCountEstimate", rowCountEstimate, ...
                "textBandScore", textBandScore, ...
                "edgeClipPenalty", edgeClipPenalty, ...
                "boundaryPenalty", boundaryPenalty, ...
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
        "layoutHint", layoutHint, ...
        "rowCountEstimate", rowCountEstimate, ...
        "emptyRegionPenalty", emptyRegionPenalty, ...
        "scoreBreakdown", bestBreakdown);
end

function score = localBranchScore(branchName, scale)
    % LOCALBRANCHSCORE Assigns a small bias/confidence score depending on 
    % which detection method generated this candidate. Edge priority is highest.

    switch string(branchName)
        case "edge_priority"
            base = 0.95;
        case "text_priority"
            base = 1.00;
        case "text_dark_priority"
            base = 1.02;
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
    % LOCALTHRESHOLDSCORE Returns a score based on whether a value is within a desired range,
    % with a smooth falloff outside the range.
    
    if value <= lowTarget
        score = max(0, value / max(lowTarget, eps));
    elseif value >= highTarget
        score = max(0, 1 - (value - highTarget) / max(1 - highTarget, eps));
    else
        score = 1;
    end
end

function score = localTextComponentCountScore(textComponentCount, profileName)
    if nargin < 2
        profileName = "single_line";
    end

    if string(profileName) == "two_row"
        targetRange = [4 9];
    else
        targetRange = [5 8];
    end

    if textComponentCount >= targetRange(1) && textComponentCount <= targetRange(2)
        score = 1.0;
    elseif textComponentCount == targetRange(1) - 1 || textComponentCount == targetRange(2) + 1
        score = 0.58;
    elseif textComponentCount == targetRange(1) - 2 || textComponentCount == targetRange(2) + 2
        score = 0.28;
    else
        score = 0.10;
    end
end

function score = localCandidatePositionScore(horizontalCenterRatio, verticalCenterRatio, config)
    horizontalScore = localCenterAxisScore( ...
        horizontalCenterRatio, ...
        config.detection.targetHorizontalCenterRatio, ...
        config.detection.minHorizontalCenterRatio, ...
        config.detection.maxHorizontalCenterRatio);
    verticalScore = localCenterAxisScore( ...
        verticalCenterRatio, ...
        config.detection.targetVerticalCenterRatio, ...
        config.detection.minVerticalCenterRatio, ...
        config.detection.maxVerticalCenterRatio);
    score = mean([horizontalScore verticalScore]);
end

function score = localCenterAxisScore(value, targetValue, minValue, maxValue)
    if value < minValue || value > maxValue
        score = 0;
        return;
    end

    span = max(maxValue - minValue, eps);
    score = max(0.15, 1 - abs(value - targetValue) / span);
end

function score = localCandidateSpanScore(widthRatio, heightRatio, aspectRatio, profile)
    widthScore = localProfileFit(widthRatio, profile.targetWidthRatio, profile.minWidthRatio, profile.maxWidthRatio);
    heightScore = localProfileFit(heightRatio, profile.targetHeightRatio, profile.minHeightRatio, profile.maxHeightRatio);
    aspectScore = localProfileFit(aspectRatio, profile.targetAspectRatio, profile.minAspectRatio, profile.maxAspectRatio);
    score = mean([widthScore heightScore aspectScore]);
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

function penalty = localSplitFragmentPenalty(candidateMask, layoutHint, profileName)
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
    if string(profileName) == "two_row" || string(layoutHint) == "two_row"
        penalty = 0.40 * penalty;
    end
end

function [score, clipPenalty] = localTextBandCompletenessScore(candidateImage, profileName)
    if nargin >= 2 && string(profileName) == "two_row"
        [score, clipPenalty] = localTwoRowTextBandScore(candidateImage);
        return;
    end

    [score, clipPenalty] = localSingleRowTextBandScore(candidateImage);
end

function [score, clipPenalty] = localSingleRowTextBandScore(candidateImage)
    score = 0.30;
    clipPenalty = 0;

    if isempty(candidateImage)
        return;
    end

    grayImage = candidateImage;
    if ndims(grayImage) == 3
        grayImage = im2gray(grayImage);
    end
    grayImage = im2single(grayImage);

    rowStart = max(1, round(0.20 * size(grayImage, 1)));
    rowEnd = min(size(grayImage, 1), round(0.86 * size(grayImage, 1)));
    bandImage = grayImage(rowStart:rowEnd, :);

    brightMask = imbinarize(bandImage, "adaptive", ...
        "ForegroundPolarity", "bright", ...
        "Sensitivity", 0.40);
    brightMask = bwareaopen(brightMask, max(4, round(numel(brightMask) * 0.002)));

    columnProfile = mean(brightMask, 1);
    if ~any(columnProfile > 0)
        return;
    end

    activeColumns = columnProfile >= max(0.03, 0.45 * mean(columnProfile(columnProfile > 0)));
    if ~any(activeColumns)
        return;
    end

    firstColumn = find(activeColumns, 1, "first");
    lastColumn = find(activeColumns, 1, "last");
    plateWidth = size(brightMask, 2);
    leftMargin = (firstColumn - 1) / max(plateWidth, 1);
    rightMargin = (plateWidth - lastColumn) / max(plateWidth, 1);
    spanRatio = (lastColumn - firstColumn + 1) / max(plateWidth, 1);

    marginScore = 0.5 * min(1, leftMargin / 0.05) + 0.5 * min(1, rightMargin / 0.05);
    symmetryScore = max(0, 1 - abs(leftMargin - rightMargin) / 0.18);
    spanScore = max(0, 1 - abs(spanRatio - 0.62) / 0.24);
    score = max(0, min(1, 0.40 * marginScore + 0.25 * symmetryScore + 0.35 * spanScore));

    leftClipPenalty = max(0, (0.025 - leftMargin) / 0.025);
    rightClipPenalty = max(0, (0.025 - rightMargin) / 0.025);
    overSpanPenalty = max(0, (spanRatio - 0.82) / 0.18);
    clipPenalty = max(0, min(1, 0.30 * leftClipPenalty + 0.50 * rightClipPenalty + 0.20 * overSpanPenalty));
end

function [score, clipPenalty] = localTwoRowTextBandScore(candidateImage)
    score = 0.22;
    clipPenalty = 0;

    if isempty(candidateImage)
        return;
    end

    grayImage = candidateImage;
    if ndims(grayImage) == 3
        grayImage = im2gray(grayImage);
    end
    grayImage = im2single(grayImage);

    brightMask = imbinarize(grayImage, "adaptive", ...
        "ForegroundPolarity", "bright", ...
        "Sensitivity", 0.42);
    brightMask = bwareaopen(brightMask, max(4, round(numel(brightMask) * 0.0018)));

    rowProfile = mean(brightMask, 2);
    positiveRows = rowProfile(rowProfile > 0);
    if isempty(positiveRows)
        return;
    end
    rowThreshold = max(0.02, 0.55 * mean(positiveRows));
    activeRows = rowProfile >= rowThreshold;
    if ~any(activeRows)
        return;
    end

    rowRuns = localLogicalRuns(activeRows);
    if size(rowRuns, 1) < 2
        [score, clipPenalty] = localSingleRowTextBandScore(candidateImage);
        score = 0.65 * score;
        return;
    end

    runHeights = rowRuns(:, 2) - rowRuns(:, 1) + 1;
    [~, order] = sort(runHeights, "descend");
    selectedRuns = sortrows(rowRuns(order(1:2), :), 1);

    topBand = brightMask(selectedRuns(1, 1):selectedRuns(1, 2), :);
    bottomBand = brightMask(selectedRuns(2, 1):selectedRuns(2, 2), :);
    [topMargins, topSpan] = localBandMargins(topBand);
    [bottomMargins, bottomSpan] = localBandMargins(bottomBand);

    rowBalanceScore = max(0, 1 - abs(runHeights(order(1)) - runHeights(order(2))) / max(sum(runHeights(order(1:2))), 1));
    separation = selectedRuns(2, 1) - selectedRuns(1, 2);
    separationScore = min(1, separation / max(0.10 * size(grayImage, 1), 1));
    spanScore = mean([ ...
        max(0, 1 - abs(topSpan - 0.58) / 0.32) ...
        max(0, 1 - abs(bottomSpan - 0.58) / 0.32)]);
    marginScore = mean([ ...
        min(1, topMargins(1) / 0.03) ...
        min(1, topMargins(2) / 0.03) ...
        min(1, bottomMargins(1) / 0.03) ...
        min(1, bottomMargins(2) / 0.03)]);

    score = max(0, min(1, mean([rowBalanceScore separationScore spanScore marginScore])));

    clipPenalty = max(0, min(1, mean([ ...
        max(0, (0.015 - topMargins(1)) / 0.015) ...
        max(0, (0.015 - topMargins(2)) / 0.015) ...
        max(0, (0.015 - bottomMargins(1)) / 0.015) ...
        max(0, (0.015 - bottomMargins(2)) / 0.015)])));
end

function penalty = localBoundaryPenalty(bbox, imageSize)
    imageHeight = imageSize(1);
    imageWidth = imageSize(2);
    leftMargin = max(0, bbox(1) - 1);
    topMargin = max(0, bbox(2) - 1);
    rightMargin = max(0, imageWidth - (bbox(1) + bbox(3)));
    bottomMargin = max(0, imageHeight - (bbox(2) + bbox(4)));

    horizontalThreshold = max(8, round(0.02 * imageWidth));
    verticalThreshold = max(8, round(0.02 * imageHeight));
    leftPenalty = max(0, (horizontalThreshold - leftMargin) / max(horizontalThreshold, 1));
    rightPenalty = max(0, (horizontalThreshold - rightMargin) / max(horizontalThreshold, 1));
    topPenalty = max(0, (verticalThreshold - topMargin) / max(verticalThreshold, 1));
    bottomPenalty = max(0, (verticalThreshold - bottomMargin) / max(verticalThreshold, 1));
    penalty = max(0, min(1, max([leftPenalty rightPenalty 0.7 * topPenalty 0.7 * bottomPenalty])));
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

function threshold = localMaskAreaThreshold(maskSize, config, scaleFactor)
    if nargin < 3
        scaleFactor = 1.0;
    end

    ratioFloor = double(config.detection.minMaskAreaRatio);
    minPixels = double(config.detection.minCandidateAreaPixels);
    threshold = max(2, round(max(minPixels * scaleFactor * 0.35, prod(maskSize(1:2)) * ratioFloor * scaleFactor)));
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
        "layoutHint", {}, ...
        "rowCountEstimate", {}, ...
        "emptyRegionPenalty", {}, ...
        "scoreBreakdown", {});
end

function score = localLayoutMatchScore(layoutHint, rowCountEstimate, profileName)
    if string(profileName) == "two_row"
        layoutScore = double(string(layoutHint) == "two_row");
        rowScore = min(1, max(0, rowCountEstimate - 1));
        score = max(0, min(1, 0.72 * layoutScore + 0.28 * rowScore));
    else
        if string(layoutHint) == "two_row" || rowCountEstimate >= 2
            score = 0.30;
        else
            score = 1.0;
        end
    end
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

function [margins, spanRatio] = localBandMargins(mask)
    if isempty(mask) || ~any(mask(:))
        margins = [0 0];
        spanRatio = 0;
        return;
    end

    columnProfile = mean(mask, 1);
    activeColumns = columnProfile > 0;
    firstColumn = find(activeColumns, 1, "first");
    lastColumn = find(activeColumns, 1, "last");
    width = size(mask, 2);
    margins = [(firstColumn - 1) / max(width, 1) (width - lastColumn) / max(width, 1)];
    spanRatio = (lastColumn - firstColumn + 1) / max(width, 1);
end

function value = localSafeFeatureValue(features, fieldName)
    value = 0;
    if isfield(features, fieldName)
        value = double(features.(fieldName));
    end
    value = max(0, min(1, value));
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
