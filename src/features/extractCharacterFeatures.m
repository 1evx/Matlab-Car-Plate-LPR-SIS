function features = extractCharacterFeatures(characterImage, config)
%EXTRACTCHARACTERFEATURES Extract handcrafted shape features from a character glyph.

if nargin < 2
    config = defaultConfig();
end

resized = imresize(im2single(characterImage), config.features.characterImageSize);
if ~islogical(resized)
    resized = imbinarize(resized);
end

if mean(resized(:)) > 0.5
    resized = ~resized;
end

downsampled = imresize(single(resized), [12 8], "nearest");
rowProjection = mean(resized, 2).';
columnProjection = mean(resized, 1);
foregroundDensity = mean(resized(:));
aspectRatio = size(resized, 2) / max(size(resized, 1), eps);
eulerValue = bweuler(resized);
componentCount = bwconncomp(resized).NumObjects;
zoneFeatures = localZoneDensities(resized, [3 2]);

features = [ ...
    downsampled(:).' ...
    rowProjection ...
    columnProjection ...
    foregroundDensity ...
    aspectRatio ...
    eulerValue ...
    componentCount ...
    zoneFeatures];
end

function zoneFeatures = localZoneDensities(binaryImage, gridSize)
rows = gridSize(1);
cols = gridSize(2);
rowEdges = round(linspace(1, size(binaryImage, 1) + 1, rows + 1));
colEdges = round(linspace(1, size(binaryImage, 2) + 1, cols + 1));
zoneFeatures = zeros(1, rows * cols);
featureIdx = 1;

for row = 1:rows
    for col = 1:cols
        rowIdx = rowEdges(row):(rowEdges(row + 1) - 1);
        colIdx = colEdges(col):(colEdges(col + 1) - 1);
        zone = binaryImage(rowIdx, colIdx);
        zoneFeatures(featureIdx) = mean(zone(:));
        featureIdx = featureIdx + 1;
    end
end
end
