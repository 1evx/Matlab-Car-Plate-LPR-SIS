function [sceneImage, metadata] = createSyntheticTwoRowPlateImage(topText, bottomText, varargin)
%CREATESYNTHETICTWOROWPLATEIMAGE Create a synthetic scene with a double-line plate.

parser = inputParser();
addRequired(parser, "topText", @(x) ischar(x) || isstring(x));
addRequired(parser, "bottomText", @(x) ischar(x) || isstring(x));
addParameter(parser, "SceneSize", [360 640 3]);
addParameter(parser, "PlatePosition", [225 170 180 110]);
addParameter(parser, "Font", "Consolas");
addParameter(parser, "TopFontSize", 24);
addParameter(parser, "BottomFontSize", 28);
parse(parser, topText, bottomText, varargin{:});

sceneSize = parser.Results.SceneSize;
plateBox = parser.Results.PlatePosition;

sceneImage = uint8(65 * ones(sceneSize));
sceneImage = localFillRectangle(sceneImage, plateBox, 255);
sceneImage = drawBoundingBoxes(sceneImage, plateBox, [0 0 0], 3);

innerLeft = plateBox(1) + 10;
innerTop = plateBox(2) + 8;
innerWidth = plateBox(3) - 20;
innerHeight = plateBox(4) - 16;
gapHeight = max(6, round(0.07 * innerHeight));
rowHeight = floor((innerHeight - gapHeight) / 2);
topRowBox = [innerLeft innerTop innerWidth rowHeight];
bottomRowBox = [innerLeft innerTop + rowHeight + gapHeight innerWidth rowHeight];

topCanvas = renderTextImage(topText, [topRowBox(4), topRowBox(3)], ...
    "FontName", parser.Results.Font, ...
    "FontSize", parser.Results.TopFontSize);
bottomCanvas = renderTextImage(bottomText, [bottomRowBox(4), bottomRowBox(3)], ...
    "FontName", parser.Results.Font, ...
    "FontSize", parser.Results.BottomFontSize);

sceneImage = localStampText(sceneImage, topCanvas, topRowBox);
sceneImage = localStampText(sceneImage, bottomCanvas, bottomRowBox);

metadata = struct( ...
    "topText", string(topText), ...
    "bottomText", string(bottomText), ...
    "plateText", append(string(topText), string(bottomText)), ...
    "plateBox", plateBox, ...
    "rowBoxes", [topRowBox; bottomRowBox]);
end

function image = localFillRectangle(image, box, grayValue)
box = round(box);
x1 = box(1);
y1 = box(2);
x2 = x1 + box(3) - 1;
y2 = y1 + box(4) - 1;
image(y1:y2, x1:x2, :) = grayValue;
end

function image = localStampText(image, textCanvas, targetBox)
textMask = imbinarize(textCanvas, "adaptive", "ForegroundPolarity", "dark");
textMask = imresize(textMask, [targetBox(4), targetBox(3)], "nearest");
rowIdx = targetBox(2):(targetBox(2) + targetBox(4) - 1);
colIdx = targetBox(1):(targetBox(1) + targetBox(3) - 1);
region = image(rowIdx, colIdx, :);

for channel = 1:3
    channelData = region(:, :, channel);
    channelData(textMask) = 0;
    region(:, :, channel) = channelData;
end

image(rowIdx, colIdx, :) = region;
end
