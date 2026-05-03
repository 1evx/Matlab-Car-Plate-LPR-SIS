function features = extractPlateFeatures(candidateMask, candidateImage)
%EXTRACTPLATEFEATURES Compute descriptive features for a plate candidate.

candidateMask = logical(candidateMask);
candidateImage = im2gray(candidateImage);
grayImage = im2double(candidateImage);
[Gmag, ~] = imgradient(grayImage);
gMean = mean(Gmag(:), "all");
% Normalized sharpness: blurry textures sit lower; strong plate strokes higher (not always separated).
focusScore = min(1, max(0, gMean / 0.11));
edgeDensity = nnz(candidateMask) / max(numel(candidateMask), 1);
stats = regionprops(candidateMask, "BoundingBox", "Extent", "Solidity");
contrastScore = std2(grayImage) / 0.22;
[characterTextureScore, plateContrastScore, alignmentScore, componentCount, ...
    emptyRegionPenalty, layoutHint, rowCountEstimate, singleLineScore, twoRowScore] = ...
    localTextEvidenceFeatures(grayImage);

if isempty(stats)
    features = struct( ...
        "edgeDensity", edgeDensity, ...
        "extent", 0, ...
        "solidity", 0, ...
        "aspectRatio", 0, ...
        "contrastScore", contrastScore, ...
        "focusScore", focusScore, ...
        "characterTextureScore", characterTextureScore, ...
        "plateContrastScore", plateContrastScore, ...
        "componentAlignmentScore", alignmentScore, ...
        "textComponentCount", componentCount, ...
        "emptyRegionPenalty", emptyRegionPenalty, ...
        "layoutHint", string(layoutHint), ...
        "rowCountEstimate", rowCountEstimate, ...
        "singleLineAlignmentScore", singleLineScore, ...
        "twoRowAlignmentScore", twoRowScore);
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
    "focusScore", focusScore, ...
    "characterTextureScore", characterTextureScore, ...
    "plateContrastScore", plateContrastScore, ...
    "componentAlignmentScore", alignmentScore, ...
    "textComponentCount", componentCount, ...
    "emptyRegionPenalty", emptyRegionPenalty, ...
    "layoutHint", string(layoutHint), ...
    "rowCountEstimate", rowCountEstimate, ...
    "singleLineAlignmentScore", singleLineScore, ...
    "twoRowAlignmentScore", twoRowScore);
end

function [textureScore, contrastScore, alignmentScore, componentCount, emptyPenalty, ...
        layoutHint, rowCountEstimate, singleLineScore, twoRowScore] = localTextEvidenceFeatures(grayImage)
localVariance = stdfilt(grayImage, true(3));
varianceScore = min(1, mean(localVariance(:)) / 0.10);
verticalGradient = mean(abs(diff(grayImage, 1, 2)), "all");
verticalStrokeScore = min(1, verticalGradient / 0.12);

[brightMask, darkMask] = localBuildTextMasks(grayImage);
brightEvidence = localLayoutEvidence(grayImage, brightMask);
darkEvidence = localLayoutEvidence(grayImage, darkMask);

brightEvidence.contrastScore = localPolarityContrast(grayImage, brightMask);
darkEvidence.contrastScore = localPolarityContrast(grayImage, darkMask);
brightTotal = 0.58 * brightEvidence.alignmentScore + 0.42 * brightEvidence.contrastScore;
darkTotal = 0.58 * darkEvidence.alignmentScore + 0.42 * darkEvidence.contrastScore;

if brightTotal >= darkTotal
    selectedEvidence = brightEvidence;
else
    selectedEvidence = darkEvidence;
end

layoutHint = selectedEvidence.layoutHint;
rowCountEstimate = selectedEvidence.rowCountEstimate;
singleLineScore = selectedEvidence.singleLineScore;
twoRowScore = selectedEvidence.twoRowScore;
alignmentScore = selectedEvidence.alignmentScore;
componentCount = selectedEvidence.componentCount;
contrastScore = selectedEvidence.contrastScore;
textureScore = max(0, min(1, mean([ ...
    varianceScore ...
    verticalStrokeScore ...
    selectedEvidence.densityScore])));
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
mask = bwareaopen(mask, max(2, round(numel(mask) * 0.0012)));
end

function evidence = localLayoutEvidence(grayImage, binaryMask)
evidence = struct( ...
    "alignmentScore", 0, ...
    "componentCount", 0, ...
    "densityScore", 0, ...
    "layoutHint", "unknown", ...
    "rowCountEstimate", 0, ...
    "singleLineScore", 0, ...
    "twoRowScore", 0, ...
    "contrastScore", 0);

if ~any(binaryMask(:))
    return;
end

boxes = localCollectCandidateBoxes(binaryMask);
componentCount = size(boxes, 1);
evidence.componentCount = componentCount;
if componentCount == 0
    return;
end

imageHeight = size(binaryMask, 1);
singleLineScore = localSingleRowScore(boxes, imageHeight);
[rowGroups, rowGap, hasTwoRows] = localSplitBoxesIntoRows(boxes, imageHeight);
if hasTwoRows
    twoRowScore = localTwoRowScore(rowGroups, boxes, imageHeight, rowGap);
    rowCountEstimate = 2;
else
    twoRowScore = 0.08 * localExpectedCountScore(componentCount, [4 9]);
    rowCountEstimate = 1;
end

twoRowPromotionMargin = 0.14;
twoRowMinScore = 0.60;
if hasTwoRows && twoRowScore >= twoRowMinScore && twoRowScore > singleLineScore + twoRowPromotionMargin
    layoutHint = "two_row";
    alignmentScore = twoRowScore;
    densityScore = localExpectedCountScore(componentCount, [4 9]);
else
    layoutHint = "single_line";
    alignmentScore = singleLineScore;
    densityScore = localExpectedCountScore(componentCount, [3 8]);
end

evidence.alignmentScore = max(0, min(1, alignmentScore));
evidence.densityScore = max(0, min(1, densityScore));
evidence.layoutHint = layoutHint;
evidence.rowCountEstimate = rowCountEstimate;
evidence.singleLineScore = max(0, min(1, singleLineScore));
evidence.twoRowScore = max(0, min(1, twoRowScore));
end

function boxes = localCollectCandidateBoxes(binaryMask)
stats = regionprops("table", binaryMask, "BoundingBox", "Area");
boxes = zeros(0, 4);
if isempty(stats)
    return;
end

imageHeight = size(binaryMask, 1);
imageWidth = size(binaryMask, 2);
imageArea = numel(binaryMask);

for i = 1:height(stats)
    box = stats.BoundingBox(i, :);
    areaRatio = stats.Area(i) / max(imageArea, eps);
    heightRatio = box(4) / max(imageHeight, eps);
    widthRatio = box(3) / max(imageWidth, eps);
    aspectRatio = box(3) / max(box(4), eps);

    if areaRatio >= 0.0007 && areaRatio <= 0.20 && ...
            heightRatio >= 0.08 && heightRatio <= 0.95 && ...
            widthRatio >= 0.008 && widthRatio <= 0.28 && ...
            aspectRatio >= 0.05 && aspectRatio <= 1.80
        boxes(end + 1, :) = box; %#ok<AGROW>
    end
end
end

function score = localSingleRowScore(boxes, imageHeight)
if isempty(boxes)
    score = 0;
    return;
end

componentCount = size(boxes, 1);
densityScore = localExpectedCountScore(componentCount, [3 8]);
centersY = boxes(:, 2) + boxes(:, 4) / 2;
heightValues = boxes(:, 4);
centerScore = max(0, 1 - std(centersY) / max(0.16 * imageHeight, eps));
heightScore = max(0, 1 - std(heightValues) / max(mean(heightValues), eps));
gapScore = localGapConsistency(boxes);
score = max(0, min(1, mean([densityScore centerScore heightScore gapScore])));
end

function [rowGroups, rowGap, hasTwoRows] = localSplitBoxesIntoRows(boxes, imageHeight)
rowGroups = cell(0, 1);
rowGap = 0;
hasTwoRows = false;

if size(boxes, 1) < 4
    return;
end

centersY = boxes(:, 2) + boxes(:, 4) / 2;
[sortedCenters, order] = sort(centersY, "ascend");
sortedBoxes = boxes(order, :);
gaps = diff(sortedCenters);
if isempty(gaps)
    return;
end

[rowGap, splitIdx] = max(gaps);
heightValues = sortedBoxes(:, 4);
gapThreshold = max(0.45 * mean(heightValues), 0.08 * imageHeight);
if rowGap < gapThreshold
    return;
end

topBoxes = sortedBoxes(1:splitIdx, :);
bottomBoxes = sortedBoxes(splitIdx + 1:end, :);
if size(topBoxes, 1) < 2 || size(bottomBoxes, 1) < 2
    return;
end

rowGroups = {topBoxes, bottomBoxes};
hasTwoRows = true;
end

function score = localTwoRowScore(rowGroups, allBoxes, imageHeight, rowGap)
if numel(rowGroups) ~= 2
    score = 0;
    return;
end

topBoxes = rowGroups{1};
bottomBoxes = rowGroups{2};
totalCount = size(allBoxes, 1);
rowCounts = [size(topBoxes, 1) size(bottomBoxes, 1)];
countScore = localExpectedCountScore(totalCount, [4 9]);
balanceScore = max(0, 1 - abs(diff(rowCounts)) / max(totalCount, 1));
topCompactness = localRowCompactness(topBoxes, imageHeight);
bottomCompactness = localRowCompactness(bottomBoxes, imageHeight);
withinRowScore = mean([topCompactness bottomCompactness]);
heightBalance = max(0, 1 - abs(mean(topBoxes(:, 4)) - mean(bottomBoxes(:, 4))) / ...
    max(mean(allBoxes(:, 4)), eps));
separationScore = min(1, rowGap / max(0.55 * mean(allBoxes(:, 4)), eps));
horizontalConsistency = localTwoRowHorizontalConsistency(rowGroups, allBoxes);
score = max(0, min(1, mean([ ...
    countScore ...
    balanceScore ...
    withinRowScore ...
    heightBalance ...
    separationScore ...
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

function unionBox = localUnionBox(boxes)
if isempty(boxes)
    unionBox = [1 1 0 0];
    return;
end

minX = min(boxes(:, 1));
minY = min(boxes(:, 2));
maxX = max(boxes(:, 1) + boxes(:, 3));
maxY = max(boxes(:, 2) + boxes(:, 4));
unionBox = [minX minY maxX - minX maxY - minY];
end
