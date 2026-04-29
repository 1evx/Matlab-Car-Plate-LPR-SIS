function [rectifiedPlate, metadata] = rectifyPlate(inputImage, bbox, config)
    %RECTIFYPLATE Crop and deskew a detected plate region.

    config = validateConfig(config);
    paddedBBox = localPadBBox(bbox, size(inputImage), config.rectification.cropPaddingRatio);
    cropped = imcrop(inputImage, paddedBBox);
    if isempty(cropped)
        rectifiedPlate = [];
        metadata = struct("angle", 0, "angleConfidence", 0, "croppedPlate", [], "binaryMask", []);
        return;
    end

    gray = im2gray(cropped);
    enhanced = enhanceContrast(reduceNoise(normalizeLighting(gray, config), config), config);
    edgeMask = edge(enhanced, "Canny");
    [angle, angleConfidence] = localEstimateSkew(edgeMask, config);

    if angleConfidence >= config.rectification.minAngleConfidence && abs(angle) > 0.5
        rectifiedPlate = imrotate(cropped, angle, "bilinear", "crop");
        rotatedEnhanced = imrotate(enhanced, angle, "bilinear", "crop");
    else
        angle = 0;
        rectifiedPlate = cropped;
        rotatedEnhanced = enhanced;
    end

    rotatedBinary = imbinarize(rotatedEnhanced, "adaptive", "ForegroundPolarity", "dark");

    metadata = struct( ...
        "angle", angle, ...
        "angleConfidence", angleConfidence, ...
        "croppedPlate", cropped, ...
        "binaryMask", rotatedBinary);
end

function [angle, confidence] = localEstimateSkew(edgeMask, config)
    angle = 0;
    confidence = 0;

    if nnz(edgeMask) == 0
        return;
    end

    [H, theta, rho] = hough(edgeMask); %#ok<ASGLU>
    peakCount = min(8, max(1, round(nnz(edgeMask) / 250)));
    peaks = houghpeaks(H, peakCount, "Threshold", ceil(0.25 * max(H(:))));
    if isempty(peaks)
        return;
    end

    lines = houghlines(edgeMask, theta, rho, peaks, ...
        "FillGap", 8, ...
        "MinLength", config.rectification.minLineLength);
    if isempty(lines)
        return;
    end

    candidateAngles = [];
    candidateWeights = [];
    for i = 1:numel(lines)
        point1 = double(lines(i).point1);
        point2 = double(lines(i).point2);
        delta = point2 - point1;
        lineLength = hypot(delta(1), delta(2));
        if lineLength < config.rectification.minLineLength
            continue;
        end

        lineAngle = atan2d(delta(2), delta(1));
        lineAngle = mod(lineAngle + 90, 180) - 90;
        if abs(lineAngle) > config.rectification.maxRotationDegrees
            continue;
        end

        candidateAngles(end+1) = lineAngle; %#ok<AGROW>
        candidateWeights(end+1) = lineLength; %#ok<AGROW>
    end

    if isempty(candidateAngles)
        return;
    end

    angle = sum(candidateAngles .* candidateWeights) / sum(candidateWeights);
    inlierMask = abs(candidateAngles - angle) <= 2.5;
    confidence = sum(candidateWeights(inlierMask)) / max(sum(candidateWeights), eps);
    angle = max(min(angle, config.rectification.maxRotationDegrees), -config.rectification.maxRotationDegrees);
end

function paddedBBox = localPadBBox(bbox, imageSize, paddingRatio)
    paddingX = bbox(3) * paddingRatio;
    paddingY = bbox(4) * paddingRatio;

    x1 = max(1, floor(bbox(1) - paddingX));
    y1 = max(1, floor(bbox(2) - paddingY));
    x2 = min(imageSize(2), ceil(bbox(1) + bbox(3) + paddingX));
    y2 = min(imageSize(1), ceil(bbox(2) + bbox(4) + paddingY));

    paddedBBox = [x1 y1 max(1, x2 - x1) max(1, y2 - y1)];
end
