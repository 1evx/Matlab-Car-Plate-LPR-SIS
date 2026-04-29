function outputImage = drawBoundingBoxes(inputImage, boxes, color, lineWidth)
%DRAWBOUNDINGBOXES Draw simple rectangle borders without Computer Vision Toolbox.

if nargin < 3 || isempty(color)
    color = [0 255 0];
end

if nargin < 4 || isempty(lineWidth)
    lineWidth = 2;
end

outputImage = inputImage;
if isempty(outputImage)
    return;
end

if ndims(outputImage) == 2
    outputImage = repmat(outputImage, 1, 1, 3);
end

boxes = reshape(boxes, [], 4);
for i = 1:size(boxes, 1)
    outputImage = localDrawSingleBox(outputImage, boxes(i, :), color, lineWidth);
end
end

function image = localDrawSingleBox(image, box, color, lineWidth)
imageSize = size(image);
box = round(box);

x1 = max(1, box(1));
y1 = max(1, box(2));
x2 = min(imageSize(2), x1 + max(1, box(3)) - 1);
y2 = min(imageSize(1), y1 + max(1, box(4)) - 1);

for offset = 0:max(0, lineWidth - 1)
    top = min(imageSize(1), y1 + offset);
    bottom = max(1, y2 - offset);
    left = min(imageSize(2), x1 + offset);
    right = max(1, x2 - offset);

    image(top, left:right, 1) = color(1);
    image(top, left:right, 2) = color(2);
    image(top, left:right, 3) = color(3);

    image(bottom, left:right, 1) = color(1);
    image(bottom, left:right, 2) = color(2);
    image(bottom, left:right, 3) = color(3);

    image(top:bottom, left, 1) = color(1);
    image(top:bottom, left, 2) = color(2);
    image(top:bottom, left, 3) = color(3);

    image(top:bottom, right, 1) = color(1);
    image(top:bottom, right, 2) = color(2);
    image(top:bottom, right, 3) = color(3);
end
end

