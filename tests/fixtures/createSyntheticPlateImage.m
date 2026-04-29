function [sceneImage, metadata] = createSyntheticPlateImage(plateText, varargin)
%CREATESYNTHETICPLATEIMAGE Create a synthetic vehicle scene with a visible plate.

parser = inputParser();
addRequired(parser, "plateText", @(x) ischar(x) || isstring(x));
addParameter(parser, "SceneSize", [360 640 3]);
addParameter(parser, "PlatePosition", [190 210 250 70]);
addParameter(parser, "Font", "Consolas");
addParameter(parser, "FontSize", 30);
parse(parser, plateText, varargin{:});

sceneSize = parser.Results.SceneSize;
plateBox = parser.Results.PlatePosition;

sceneImage = uint8(65 * ones(sceneSize));
sceneImage = localFillRectangle(sceneImage, plateBox, 255);
sceneImage = drawBoundingBoxes(sceneImage, plateBox, [0 0 0], 3);

textCanvas = renderTextImage(plateText, [plateBox(4) - 10, plateBox(3) - 20], ...
    "FontName", parser.Results.Font, ...
    "FontSize", parser.Results.FontSize);
textMask = imbinarize(textCanvas, "adaptive", "ForegroundPolarity", "dark");
textMask = imresize(textMask, [plateBox(4) - 10, plateBox(3) - 20], "nearest");
textRegion = sceneImage(plateBox(2) + 5:plateBox(2) + plateBox(4) - 6, ...
    plateBox(1) + 10:plateBox(1) + plateBox(3) - 11, :);

for channel = 1:3
    channelData = textRegion(:, :, channel);
    channelData(textMask) = 0;
    textRegion(:, :, channel) = channelData;
end

sceneImage(plateBox(2) + 5:plateBox(2) + plateBox(4) - 6, ...
    plateBox(1) + 10:plateBox(1) + plateBox(3) - 11, :) = textRegion;

metadata = struct( ...
    "plateText", string(plateText), ...
    "plateBox", plateBox);
end

function image = localFillRectangle(image, box, grayValue)
box = round(box);
x1 = box(1);
y1 = box(2);
x2 = x1 + box(3) - 1;
y2 = y1 + box(4) - 1;
image(y1:y2, x1:x2, :) = grayValue;
end
