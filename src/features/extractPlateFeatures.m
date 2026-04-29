function features = extractPlateFeatures(candidateMask, candidateImage)
%EXTRACTPLATEFEATURES Compute descriptive features for a plate candidate.

candidateMask = logical(candidateMask);
candidateImage = im2gray(candidateImage);
grayImage = im2double(candidateImage);
edgeDensity = nnz(candidateMask) / numel(candidateMask);
stats = regionprops(candidateMask, "BoundingBox", "Extent", "Solidity");
contrastScore = std2(grayImage) / 0.22;
[characterTextureScore, plateContrastScore, alignmentScore, componentCount, emptyRegionPenalty] = ...
    localTextEvidenceFeatures(grayImage);

if isempty(stats)
    features = struct( ...
        "edgeDensity", edgeDensity, ...
        "extent", 0, ...
        "solidity", 0, ...
        "aspectRatio", 0, ...
        "contrastScore", contrastScore, ...
        "characterTextureScore", characterTextureScore, ...
        "plateContrastScore", plateContrastScore, ...
        "componentAlignmentScore", alignmentScore, ...
        "textComponentCount", componentCount, ...
        "emptyRegionPenalty", emptyRegionPenalty);
    return;
end

bbox = stats(1).BoundingBox;
aspectRatio = bbox(3) / max(bbox(4), eps);
features = struct( ...
    "edgeDensity", edgeDensity, ...
    "extent", stats(1).Extent, ...
    "solidity", stats(1).Solidity, ...
    "aspectRatio", aspectRatio, ...
    "meanIntensity", mean(grayImage(:)), ...
    "contrastScore", contrastScore, ...
    "characterTextureScore", characterTextureScore, ...
    "plateContrastScore", plateContrastScore, ...
    "componentAlignmentScore", alignmentScore, ...
    "textComponentCount", componentCount, ...
    "emptyRegionPenalty", emptyRegionPenalty);
end

function [textureScore, contrastScore, alignmentScore, componentCount, emptyPenalty] = localTextEvidenceFeatures(grayImage)
localVariance = stdfilt(grayImage, true(3));
varianceScore = min(1, mean(localVariance(:)) / 0.10);
verticalGradient = mean(abs(diff(grayImage, 1, 2)), "all");
verticalStrokeScore = min(1, verticalGradient / 0.12);

[brightMask, darkMask] = localBuildTextMasks(grayImage);
[brightAlignment, brightCount, brightDensity] = localAlignmentForMask(brightMask);
[darkAlignment, darkCount, darkDensity] = localAlignmentForMask(darkMask);

if brightAlignment >= darkAlignment
    alignmentScore = brightAlignment;
    componentCount = brightCount;
    componentDensityScore = brightDensity;
    contrastScore = localPolarityContrast(grayImage, brightMask);
else
    alignmentScore = darkAlignment;
    componentCount = darkCount;
    componentDensityScore = darkDensity;
    contrastScore = localPolarityContrast(grayImage, darkMask);
end

textureScore = max(0, min(1, mean([varianceScore verticalStrokeScore componentDensityScore])));
emptyPenalty = max(0, min(1, 1 - mean([textureScore contrastScore alignmentScore])));
end

function [brightMask, darkMask] = localBuildTextMasks(grayImage)
brightMask = imbinarize(grayImage, "adaptive", "ForegroundPolarity", "bright", "Sensitivity", 0.45);
darkMask = imbinarize(grayImage, "adaptive", "ForegroundPolarity", "dark", "Sensitivity", 0.45);

brightMask = localCleanupTextMask(brightMask);
darkMask = localCleanupTextMask(darkMask);
end

function mask = localCleanupTextMask(mask)
mask = logical(mask);
mask = imopen(mask, strel("rectangle", [2 2]));
mask = imclose(mask, strel("rectangle", [2 3]));
mask = imclearborder(mask);
mask = bwareaopen(mask, 6);
end

function [score, componentCount, densityScore] = localAlignmentForMask(binaryMask)
componentCount = 0;
densityScore = 0;
score = 0;

if ~any(binaryMask(:))
    return;
end

stats = regionprops("table", binaryMask, "BoundingBox", "Area");
if isempty(stats)
    return;
end

imageHeight = size(binaryMask, 1);
imageWidth = size(binaryMask, 2);
imageArea = numel(binaryMask);
boxes = zeros(0, 4);

for i = 1:height(stats)
    box = stats.BoundingBox(i, :);
    areaRatio = stats.Area(i) / max(imageArea, eps);
    heightRatio = box(4) / max(imageHeight, eps);
    widthRatio = box(3) / max(imageWidth, eps);
    aspectRatio = box(3) / max(box(4), eps);

    if areaRatio >= 0.002 && areaRatio <= 0.18 && ...
            heightRatio >= 0.18 && heightRatio <= 0.92 && ...
            widthRatio >= 0.01 && widthRatio <= 0.24 && ...
            aspectRatio >= 0.08 && aspectRatio <= 1.35
        boxes(end+1, :) = box; %#ok<AGROW>
    end
end

componentCount = size(boxes, 1);
if componentCount == 0
    return;
end

densityScore = localExpectedCountScore(componentCount, [3 8]);
centersY = boxes(:, 2) + boxes(:, 4) / 2;
heightValues = boxes(:, 4);
centerScore = max(0, 1 - std(centersY) / max(0.18 * imageHeight, eps));
heightScore = max(0, 1 - std(heightValues) / max(mean(heightValues), eps));
gapScore = localGapConsistency(boxes);
score = max(0, min(1, mean([densityScore centerScore heightScore gapScore])));
end

function score = localPolarityContrast(grayImage, foregroundMask)
foregroundMask = logical(foregroundMask);
if ~any(foregroundMask(:)) || all(foregroundMask(:))
    score = 0;
    return;
end

foregroundRatio = nnz(foregroundMask) / max(numel(foregroundMask), 1);
foregroundValues = grayImage(foregroundMask);
backgroundValues = grayImage(~foregroundMask);
contrastDelta = abs(mean(foregroundValues) - mean(backgroundValues));
contrastTerm = min(1, contrastDelta / 0.28);
occupancyTerm = max(0, 1 - abs(foregroundRatio - 0.28) / 0.28);
score = max(0, min(1, 0.7 * contrastTerm + 0.3 * occupancyTerm));
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
