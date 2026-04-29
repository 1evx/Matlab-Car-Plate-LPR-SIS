function estimate = estimateVehicleTypeHeuristic(inputImage, plateBBox, config)
    %ESTIMATEVEHICLETYPEHEURISTIC Return a low-risk, rule-based vehicle type estimate.
    %   This heuristic is optional and must not affect plate recognition or state ID.

    if nargin < 3
        config = defaultConfig();
    end

    config = validateConfig(config);
    estimate = localUnknownEstimate();

    if isempty(inputImage) || isempty(plateBBox)
        estimate.reason = "Plate not available for heuristic estimation.";
        return;
    end

    imageSize = size(inputImage);
    imageHeight = imageSize(1);
    imageWidth = imageSize(2);
    plateAreaRatio = (plateBBox(3) * plateBBox(4)) / max(imageHeight * imageWidth, eps);
    plateAspectRatio = plateBBox(3) / max(plateBBox(4), eps);
    plateHeightRatio = plateBBox(4) / max(imageHeight, eps);
    verticalCenterRatio = (plateBBox(2) + (plateBBox(4) / 2)) / max(imageHeight, eps);

    vehicleROI = localExpandBox(plateBBox, imageSize, config.heuristic.vehicleRoiScale);
    vehicleGray = im2gray(imcrop(inputImage, vehicleROI));
    vehicleEdgeMask = edge(vehicleGray, "Canny");
    vehicleEdgeDensity = nnz(vehicleEdgeMask) / max(numel(vehicleEdgeMask), 1);

    carScore = localCloseness(plateAreaRatio, 0.025, 0.020) * 0.40 + ...
        localCloseness(plateAspectRatio, 4.2, 2.0) * 0.35 + ...
        localCloseness(verticalCenterRatio, 0.63, 0.22) * 0.25;

    motorcycleScore = localMembership(plateAreaRatio, 0.002, 0.014) * 0.45 + ...
        localCloseness(plateAspectRatio, 2.1, 1.1) * 0.30 + ...
        localCloseness(verticalCenterRatio, 0.70, 0.18) * 0.25;

    busScore = localMembership(plateAreaRatio, 0.040, 0.180) * 0.35 + ...
        localMembership(plateHeightRatio, 0.080, 0.220) * 0.30 + ...
        localMembership(vehicleEdgeDensity, 0.060, 0.220) * 0.20 + ...
        localCloseness(verticalCenterRatio, 0.55, 0.28) * 0.15;

    scoreVector = [carScore motorcycleScore busScore];
    labelVector = ["Car / Standard vehicle" "Motorcycle-like" "Bus/large vehicle-like"];
    reasonVector = [
        "Plate size and position are consistent with a standard passenger vehicle."
        "Plate appears relatively small and compact compared with the full image."
        "Plate appears relatively large, tall, or surrounded by a larger edge-dense region."
        ];

    [bestScore, bestIndex] = max(scoreVector);

    if bestScore < config.heuristic.minConfidence
        estimate.reason = "Heuristic confidence is too low, so the result is reported as Unknown.";
    else
        estimate.label = labelVector(bestIndex);
        estimate.confidence = bestScore;
        estimate.reason = reasonVector(bestIndex);
    end

    estimate.metrics = struct( ...
        "plateAreaRatio", plateAreaRatio, ...
        "plateAspectRatio", plateAspectRatio, ...
        "plateHeightRatio", plateHeightRatio, ...
        "verticalCenterRatio", verticalCenterRatio, ...
        "vehicleEdgeDensity", vehicleEdgeDensity);
    end

    function estimate = localUnknownEstimate()
    estimate = struct( ...
        "label", "Unknown", ...
        "confidence", 0, ...
        "reason", "Not enough information for a reliable heuristic estimate.", ...
        "metrics", struct());
    end

    function score = localCloseness(value, centerValue, tolerance)
    score = max(0, 1 - (abs(value - centerValue) / max(tolerance, eps)));
    end

    function score = localMembership(value, minValue, maxValue)
    if value < minValue || value > maxValue
        score = 0;
    elseif value <= (minValue + maxValue) / 2
        score = (value - minValue) / max(((minValue + maxValue) / 2) - minValue, eps);
    else
        score = (maxValue - value) / max(maxValue - ((minValue + maxValue) / 2), eps);
    end
    end

    function bbox = localExpandBox(bbox, imageSize, scaleFactor)
    centerX = bbox(1) + (bbox(3) / 2);
    centerY = bbox(2) + (bbox(4) / 2);
    newWidth = bbox(3) * scaleFactor;
    newHeight = bbox(4) * scaleFactor;

    x1 = max(1, floor(centerX - (newWidth / 2)));
    y1 = max(1, floor(centerY - (newHeight / 2)));
    x2 = min(imageSize(2), ceil(centerX + (newWidth / 2)));
    y2 = min(imageSize(1), ceil(centerY + (newHeight / 2)));

    bbox = [x1 y1 max(1, x2 - x1) max(1, y2 - y1)];
end
