function [sceneImage, metadata] = createSyntheticTinyPlateImage(plateText, varargin)
%CREATESYNTHETICTINYPLATEIMAGE Create a scene with a small, far-looking plate.

parser = inputParser();
addRequired(parser, "plateText", @(x) ischar(x) || isstring(x));
addParameter(parser, "SceneSize", [360 640 3]);
addParameter(parser, "PlatePosition", [445 215 92 28]);
addParameter(parser, "Font", "Consolas");
addParameter(parser, "FontSize", 13);
parse(parser, plateText, varargin{:});

[sceneImage, metadata] = createSyntheticPlateImage(plateText, ...
    "SceneSize", parser.Results.SceneSize, ...
    "PlatePosition", parser.Results.PlatePosition, ...
    "Font", parser.Results.Font, ...
    "FontSize", parser.Results.FontSize);

sceneImage = imgaussfilt(sceneImage, 0.35);
metadata.syntheticScale = "tiny";
end
