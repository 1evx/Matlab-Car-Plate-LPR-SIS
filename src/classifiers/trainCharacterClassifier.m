function model = trainCharacterClassifier(outputModelPath, config)
%TRAINCHARACTERCLASSIFIER Train a baseline character recognizer from synthetic glyphs.

if nargin < 1 || strlength(string(outputModelPath)) == 0
    cfg = defaultConfig();
    outputModelPath = cfg.classification.trainedModelPath;
else
    cfg = defaultConfig();
end

if nargin >= 2 && ~isempty(config)
    cfg = validateConfig(config);
end

labels = char([double('A'):double('Z') double('0'):double('9')]);
fonts = cfg.classification.syntheticFonts;
angles = cfg.classification.syntheticRotationDegrees;
canvasSize = cfg.classification.syntheticCanvasSize;

featureMatrix = [];
labelVector = strings(0,1);

for i = 1:numel(labels)
    for f = 1:numel(fonts)
        for a = 1:numel(angles)
            rendered = renderTextImage(labels(i), canvasSize, ...
                "FontName", fonts{f}, ...
                "FontSize", cfg.classification.syntheticFontSize);
            rendered = imrotate(rendered, angles(a), "bilinear", "crop");
            binary = imbinarize(rendered, "adaptive", "ForegroundPolarity", "dark");
            binary = bwareaopen(binary, 10);
            glyph = localTightCrop(binary);
            featureVector = extractCharacterFeatures(glyph, cfg);
            featureMatrix(end+1, :) = featureVector; %#ok<AGROW>
            labelVector(end+1, 1) = string(labels(i)); %#ok<AGROW>
        end
    end
end

model = struct();
model.labels = labelVector;
model.features = featureMatrix;
model.classifierType = "prototype";
model.createdAt = datetime("now");
model.configSnapshot = cfg.features;

if exist("fitcknn", "file") == 2
    classifier = fitcknn(featureMatrix, labelVector, ...
        "NumNeighbors", cfg.classification.kNeighbors, ...
        "Standardize", true, ...
        "Distance", "euclidean");
    model.classifier = classifier;
    model.classifierType = "fitcknn";
end

outputDir = fileparts(outputModelPath);
if ~isfolder(outputDir)
    mkdir(outputDir);
end
save(outputModelPath, "model");

logger("info", "Saved baseline character model to %s", outputModelPath);
end

function glyph = localTightCrop(binaryImage)
props = regionprops(binaryImage, "BoundingBox", "Area");
if isempty(props)
    glyph = binaryImage;
    return;
end

[~, idx] = max([props.Area]);
bbox = ceil(props(idx).BoundingBox);
glyph = imcrop(binaryImage, bbox);
if isempty(glyph)
    glyph = binaryImage;
end
end
