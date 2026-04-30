function [recognizedText, metadata] = recognizeCharacters(plateImage, config)
    %RECOGNIZECHARACTERS Recognize plate text from a rectified plate image using MATLAB OCR.

    if nargin < 2
        config = defaultConfig();
    end

    config = validateConfig(config);
    metadata = localEmptyMetadata(plateImage);

    if isempty(plateImage)
        recognizedText = "";
        return;
    end

    variants = localOcrVariants(plateImage, config);
    bestResult = struct("text", "", "confidence", 0, "success", false, "errorMessage", "");
    bestVariantName = "";
    bestVariantImage = plateImage;
    attemptedResults = cell(numel(variants), 1);
    bestScore = -inf;

    for i = 1:numel(variants)
        variantConfig = config;
        variantConfig.classification.matlabOcrTextLayout = variants(i).layout;
        ocrResult = runMatlabOCR(variants(i).image, variantConfig);
        attemptedResults{i} = struct( ...
            "name", variants(i).name, ...
            "layout", string(variants(i).layout), ...
            "image", variants(i).image, ...
            "result", ocrResult);

        candidateText = localNormalizeOcrText(ocrResult.text);
        candidateText = localCanonicalizeOcrText(candidateText, config);
        candidateScore = localOcrCandidateScore(candidateText, ocrResult.confidence, ocrResult.success, config);
        if candidateScore > bestScore
            bestScore = candidateScore;
            bestResult = ocrResult;
            bestResult.text = candidateText;
            bestVariantName = variants(i).name;
            bestVariantImage = variants(i).image;
        end
    end

    recognizedText = string(bestResult.text);
    predictions = localStringCharacters(recognizedText);
    confidences = localExpandOcrConfidence(recognizedText, bestResult);

    metadata = struct( ...
        "predictions", predictions, ...
        "confidences", confidences, ...
        "rawPredictions", predictions, ...
        "rawConfidences", confidences, ...
        "stateLookupText", string(recognizedText), ...
        "parsedText", string(recognizedText), ...
        "parsedPattern", localOcrPatternLabel(recognizedText, config), ...
        "topCandidates", {localOcrTopCandidates(predictions, confidences)}, ...
        "method", "matlab_ocr", ...
        "plateImage", plateImage, ...
        "ocrInputPlate", bestVariantImage, ...
        "ocrInputName", string(bestVariantName), ...
        "attemptedResults", {attemptedResults}, ...
        "matlabOcr", bestResult);
end

function variants = localOcrVariants(plateImage, config)
    grayImage = plateImage;
    if ndims(grayImage) == 3
        grayImage = im2gray(grayImage);
    end

    preparedImage = prepareTesseractPlateImage(plateImage, config);
    invgrayImage = imcomplement(grayImage);
    binaryImage = imbinarize(im2single(grayImage), "adaptive", ...
        "ForegroundPolarity", "bright", ...
        "Sensitivity", 0.42);
    binaryImage = uint8(binaryImage) * 255;

    variants = [ ...
        struct("name", "raw_line", "layout", "Line", "image", plateImage); ...
        struct("name", "gray_line", "layout", "Line", "image", grayImage); ...
        struct("name", "binary_line", "layout", "Line", "image", binaryImage); ...
        struct("name", "invgray_line", "layout", "Line", "image", invgrayImage); ...
        struct("name", "raw_block", "layout", "Block", "image", plateImage); ...
        struct("name", "gray_block", "layout", "Block", "image", grayImage); ...
        struct("name", "invgray_block", "layout", "Block", "image", invgrayImage); ...
        struct("name", "prepared_block", "layout", "Block", "image", preparedImage); ...
        struct("name", "prepared_word", "layout", "Word", "image", preparedImage) ...
    ];
end

function score = localOcrCandidateScore(text, confidence, success, config)
    text = string(text);
    if ~logical(success) || strlength(text) == 0
        score = -1;
        return;
    end

    normalized = upper(regexprep(text, "[^A-Z0-9]", ""));
    letterCount = strlength(regexprep(normalized, "[^A-Z]", ""));
    digitCount = strlength(regexprep(normalized, "[^0-9]", ""));
    totalCount = strlength(normalized);
    [prefixLength, leadingDigitCount, suffixLength, isOrdered] = localOcrTextShape(normalized);

    allowSuffixLetter = isfield(config, "classification") && ...
        isfield(config.classification, "allowSuffixLetter") && ...
        logical(config.classification.allowSuffixLetter);
    if allowSuffixLetter
        exactPattern = "^[A-Z]{1,4}[0-9]{1,4}[A-Z]?$";
    else
        exactPattern = "^[A-Z]{1,4}[0-9]{1,4}$";
    end

    regexBonus = 0;
    if ~isempty(regexp(normalized, exactPattern, "once"))
        regexBonus = 0.40;
    elseif ~isempty(regexp(normalized, "^[A-Z]{1,4}[0-9]{1,4}", "once"))
        regexBonus = 0.22;
    end

    compositionBonus = 0;
    if letterCount >= 1 && digitCount >= 1
        compositionBonus = 0.18;
    end

    structureBonus = 0;
    if isOrdered
        structureBonus = structureBonus + 0.12;
    end
    if prefixLength >= 2 && prefixLength <= 3
        structureBonus = structureBonus + 0.16;
    elseif prefixLength == 1
        structureBonus = structureBonus + 0.04;
    end
    if leadingDigitCount >= 3 && leadingDigitCount <= 4
        structureBonus = structureBonus + 0.10;
    elseif leadingDigitCount >= 1
        structureBonus = structureBonus + 0.03;
    end
    if suffixLength > 0
        if allowSuffixLetter
            structureBonus = structureBonus - 0.06 * suffixLength;
        else
            structureBonus = structureBonus - 0.24 * suffixLength;
        end
    end

    lengthPenalty = 0.10 * abs(totalCount - 7);
    shortPenalty = 0;
    if totalCount < 5
        shortPenalty = shortPenalty + 0.18;
    end
    if prefixLength <= 1 && totalCount <= 5
        shortPenalty = shortPenalty + 0.16;
    end
    if prefixLength == 0 && digitCount > 0
        shortPenalty = shortPenalty + 0.12;
    end

    exactNoSuffixBonus = 0;
    if suffixLength == 0 && isOrdered && prefixLength >= 1 && prefixLength <= 4 && ...
            leadingDigitCount >= 1 && leadingDigitCount <= 4
        exactNoSuffixBonus = 0.08;
    end

    forbiddenSuffixPenalty = 0;
    if suffixLength > 0 && ~allowSuffixLetter
        forbiddenSuffixPenalty = 0.16 + 0.08 * min(suffixLength, 2);
    end

    score = double(confidence) + regexBonus + compositionBonus + structureBonus + exactNoSuffixBonus - ...
        lengthPenalty - shortPenalty - forbiddenSuffixPenalty;
end

function [prefixLength, digitLength, suffixLength, isOrdered] = localOcrTextShape(normalized)
    prefixMatch = regexp(char(normalized), "^[A-Z]+", "match", "once");
    if isempty(prefixMatch)
        prefixLength = 0;
    else
        prefixLength = strlength(string(prefixMatch));
    end

    remaining = extractAfter(normalized, prefixLength);
    digitMatch = regexp(char(remaining), "^[0-9]+", "match", "once");
    if isempty(digitMatch)
        digitLength = 0;
    else
        digitLength = strlength(string(digitMatch));
    end

    suffix = extractAfter(remaining, digitLength);
    suffixLength = strlength(regexprep(suffix, "[^A-Z]", ""));
    isOrdered = (prefixLength + digitLength + suffixLength) == strlength(normalized) && ...
        suffixLength == strlength(suffix);
end

function metadata = localEmptyMetadata(plateImage)
    metadata = struct( ...
        "predictions", strings(0, 1), ...
        "confidences", zeros(0, 1), ...
        "rawPredictions", strings(0, 1), ...
        "rawConfidences", zeros(0, 1), ...
        "stateLookupText", "", ...
        "parsedText", "", ...
        "parsedPattern", "", ...
        "topCandidates", {cell(0, 1)}, ...
        "method", "matlab_ocr", ...
        "plateImage", plateImage, ...
        "ocrInputPlate", plateImage, ...
        "ocrInputName", "", ...
        "attemptedResults", {cell(0, 1)}, ...
        "matlabOcr", struct( ...
            "success", false, ...
            "text", "", ...
            "confidence", 0, ...
            "words", strings(0, 1), ...
            "wordConfidences", zeros(0, 1), ...
            "wordBoundingBoxes", zeros(0, 4), ...
            "characterConfidences", zeros(0, 1), ...
            "characterBoundingBoxes", zeros(0, 4), ...
            "errorMessage", ""));
end

function normalizedText = localNormalizeOcrText(text)
    normalizedText = upper(regexprep(string(text), "[^A-Z0-9]", ""));
end

function canonicalText = localCanonicalizeOcrText(text, config)
    canonicalText = upper(regexprep(string(text), "[^A-Z0-9]", ""));
    allowSuffixLetter = isfield(config, "classification") && ...
        isfield(config.classification, "allowSuffixLetter") && ...
        logical(config.classification.allowSuffixLetter);
    if allowSuffixLetter || strlength(canonicalText) == 0
        return;
    end

    suffixMatch = regexp(char(canonicalText), "^([A-Z]{1,4}[0-9]{1,4})[A-Z]$", "tokens", "once");
    if ~isempty(suffixMatch)
        canonicalText = string(suffixMatch{1});
    end
end

function confidences = localExpandOcrConfidence(recognizedText, ocrResult)
    count = strlength(string(recognizedText));
    if count == 0
        confidences = zeros(0, 1);
        return;
    end

    if isfield(ocrResult, "characterConfidences") && numel(ocrResult.characterConfidences) == count
        confidences = max(0, min(1, double(ocrResult.characterConfidences(:))));
        return;
    end

    confidences = repmat(max(0, min(1, double(ocrResult.confidence))), count, 1);
end

function predictions = localStringCharacters(text)
    if strlength(string(text)) == 0
        predictions = strings(0, 1);
        return;
    end
    predictions = string(regexp(char(string(text)), ".", "match")).';
end

function topCandidates = localOcrTopCandidates(predictions, confidences)
    topCandidates = cell(numel(predictions), 1);
    for i = 1:numel(predictions)
        topCandidates{i} = struct( ...
            "labels", predictions(i), ...
            "scores", confidences(i));
    end
end

function patternLabel = localOcrPatternLabel(recognizedText, config)
    normalized = string(recognizedText);
    if strlength(normalized) == 0
        patternLabel = "";
        return;
    end

    digitStart = regexp(char(normalized), "[0-9]", "once");
    if isempty(digitStart)
        prefixLength = strlength(regexprep(normalized, "[^A-Z]", ""));
    else
        prefixLength = strlength(regexprep(extractBefore(normalized, digitStart), "[^A-Z]", ""));
    end
    digitRun = regexp(char(normalized), "[0-9]+", "match", "once");
    if isempty(digitRun)
        digitLength = 0;
    else
        digitLength = strlength(string(digitRun));
    end
    suffixLength = strlength(normalized) - prefixLength - digitLength;

    if prefixLength >= min(config.classification.allowedPrefixLengths) && ...
            prefixLength <= max(config.classification.allowedPrefixLengths) && ...
            digitLength >= min(config.classification.allowedDigitLengths) && ...
            digitLength <= max(config.classification.allowedDigitLengths)
        patternLabel = sprintf("L%d-D%d-S%d", prefixLength, digitLength, suffixLength);
    else
        patternLabel = "unparsed";
    end
end
