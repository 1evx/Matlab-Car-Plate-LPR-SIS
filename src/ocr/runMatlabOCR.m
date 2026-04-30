function result = runMatlabOCR(plateImage, config)
    %RUNMATLABOCR Run MATLAB's built-in OCR on a plate image.

    result = struct( ...
        "success", false, ...
        "text", "", ...
        "confidence", 0, ...
        "words", strings(0, 1), ...
        "wordConfidences", zeros(0, 1), ...
        "wordBoundingBoxes", zeros(0, 4), ...
        "characterConfidences", zeros(0, 1), ...
        "characterBoundingBoxes", zeros(0, 4), ...
        "errorMessage", "");

    if isempty(plateImage)
        result.errorMessage = "Empty OCR image.";
        return;
    end

    if ~(exist("ocr", "file") == 2 || exist("ocr", "builtin") == 5)
        result.errorMessage = "MATLAB OCR function is unavailable.";
        return;
    end

    layoutValue = localLayoutAnalysisValue(config);
    characterSet = char(string(config.classification.matlabOcrCharacterSet));

    try
        ocrOutput = localRunCompatibleOcr(plateImage, layoutValue, characterSet);
    catch exception
        result.errorMessage = string(exception.message);
        return;
    end

    result.text = string(strtrim(ocrOutput.Text));
    result.words = localReadOcrProperty(ocrOutput, "Words", strings(0, 1));
    result.wordConfidences = localClampConfidence(localReadOcrProperty(ocrOutput, "WordConfidences", zeros(0, 1)));
    result.wordBoundingBoxes = localReadOcrProperty(ocrOutput, "WordBoundingBoxes", zeros(0, 4));
    result.characterConfidences = localClampConfidence(localReadOcrProperty(ocrOutput, "CharacterConfidences", zeros(0, 1)));
    result.characterBoundingBoxes = localReadOcrProperty(ocrOutput, "CharacterBoundingBoxes", zeros(0, 4));

    if ~isempty(result.wordConfidences)
        result.confidence = mean(result.wordConfidences);
    elseif ~isempty(result.characterConfidences)
        result.confidence = mean(result.characterConfidences);
    else
        result.confidence = 0;
    end

    result.success = strlength(result.text) > 0;
end

function ocrOutput = localRunCompatibleOcr(plateImage, layoutValue, characterSet)
    % Use the modern LayoutAnalysis API on R2023a+ and fall back to the
    % legacy TextLayout argument for older MATLAB releases.
    try
        ocrOutput = ocr(plateImage, ...
            "LayoutAnalysis", layoutValue, ...
            "CharacterSet", characterSet);
    catch exception
        if ~localIsLegacyTextLayoutError(exception)
            rethrow(exception);
        end

        ocrOutput = ocr(plateImage, ...
            "TextLayout", localLegacyTextLayoutValue(layoutValue), ...
            "CharacterSet", characterSet);
    end
end

function layoutValue = localLayoutAnalysisValue(config)
    layoutValue = "line";
    if isfield(config, "classification") && isfield(config.classification, "matlabOcrLayoutAnalysis")
        layoutValue = lower(strtrim(string(config.classification.matlabOcrLayoutAnalysis)));
    elseif isfield(config, "classification") && isfield(config.classification, "matlabOcrTextLayout")
        layoutValue = lower(strtrim(string(config.classification.matlabOcrTextLayout)));
    end

    validValues = ["auto", "page", "block", "line", "word", "character", "none"];
    if ~any(layoutValue == validValues)
        layoutValue = "line";
    end
end

function legacyValue = localLegacyTextLayoutValue(layoutValue)
    switch string(layoutValue)
        case "auto"
            legacyValue = "Auto";
        case "page"
            legacyValue = "Paragraph";
        case "block"
            legacyValue = "Block";
        case "line"
            legacyValue = "Line";
        case "word"
            legacyValue = "Word";
        case "character"
            legacyValue = "Character";
        case "none"
            legacyValue = "Line";
        otherwise
            legacyValue = "Line";
    end
end

function isLegacyError = localIsLegacyTextLayoutError(exception)
    message = lower(string(exception.message));
    isLegacyError = contains(message, "layoutanalysis") || ...
        contains(message, "invalid parameter name") || ...
        contains(message, "name-value");
end

function value = localReadOcrProperty(ocrOutput, propertyName, defaultValue)
    value = defaultValue;
    if isprop(ocrOutput, propertyName)
        value = ocrOutput.(propertyName);
        if isstring(defaultValue)
            value = string(value);
        elseif isnumeric(defaultValue)
            value = double(value);
        end
    end
end

function confidence = localClampConfidence(confidence)
    if isempty(confidence)
        confidence = zeros(0, 1);
        return;
    end
    confidence = double(confidence);
    if max(confidence, [], "all") > 1
        confidence = confidence ./ 100;
    end
    confidence = max(0, min(1, confidence(:)));
end
