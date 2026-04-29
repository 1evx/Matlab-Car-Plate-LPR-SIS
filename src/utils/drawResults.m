function overlayImage = drawResults(inputImage, result)
%DRAWRESULTS Draw detected plate bounds and summary text on an image.

overlayImage = inputImage;
if isempty(inputImage)
    return;
end

if isfield(result, "plateBBox") && ~isempty(result.plateBBox)
    overlayImage = drawBoundingBoxes(overlayImage, result.plateBBox, [0 255 0], 3);
end
end
