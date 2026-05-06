function [recognizedText, metadata] = recognizeCharacters(plateImage, config, hints)
    %RECOGNIZECHARACTERS Recognize plate text from a rectified plate image using MATLAB OCR.

    if nargin < 2
        config = defaultConfig();
    end
    if nargin < 3 || isempty(hints)
        hints = struct();
    end

    config = validateConfig(config);
    metadata = localEmptyMetadata(plateImage, hints);

    if isempty(plateImage)
        recognizedText = "";
        return;
    end

    variants = localOcrVariants(plateImage, config, hints);
    bestResult = struct("text", "", "confidence", 0, "success", false, "errorMessage", "");
    bestVariantName = "";
    bestVariantImage = plateImage;
    attemptedResults = cell(numel(variants), 1);
    bestScore = -inf;

    for i = 1:numel(variants)
        variantConfig = config;
        variantConfig.classification.matlabOcrLayoutAnalysis = lower(string(variants(i).layout));
        variantConfig.classification.matlabOcrTextLayout = variants(i).layout;
        ocrResult = runMatlabOCR(variants(i).image, variantConfig);
        attemptedResults{i} = struct( ...
            "name", variants(i).name, ...
            "layout", string(variants(i).layout), ...
            "source", string(variants(i).source), ...
            "image", variants(i).image, ...
            "result", ocrResult);

        candidateText = localNormalizeOcrText(ocrResult.text);
        candidateText = localCanonicalizeOcrText(candidateText, config);
        candidateScore = localOcrCandidateScore(candidateText, ocrResult.confidence, ocrResult.success, config) + ...
            localVariantPreference(variants(i), hints);
        if candidateScore > bestScore
            bestScore = candidateScore;
            bestResult = ocrResult;
            bestResult.text = candidateText;
            bestVariantName = variants(i).name;
            bestVariantImage = variants(i).image;
        end
    end

    [rowwiseResult, rowwiseAttempt] = localRowwiseOcrCandidate(config, hints);
    if ~isempty(rowwiseAttempt)
        attemptedResults{end + 1} = rowwiseAttempt;
        rowwiseText = localNormalizeOcrText(rowwiseResult.text);
        rowwiseText = localCanonicalizeOcrText(rowwiseText, config);
        rowwiseScore = localOcrCandidateScore(rowwiseText, rowwiseResult.confidence, rowwiseResult.success, config) + ...
            localRowwisePreference(hints);
        if rowwiseScore > bestScore
            bestScore = rowwiseScore;
            bestResult = rowwiseResult;
            bestResult.text = rowwiseText;
            bestVariantName = "rowwise_combo";
            bestVariantImage = localPreferredOcrPreview(plateImage, hints);
        end

        [hybridResult, hybridAttempt] = localHybridTwoRowCandidate(bestResult, rowwiseResult, config, hints);
        if ~isempty(hybridAttempt)
            attemptedResults{end + 1} = hybridAttempt;
            hybridText = localNormalizeOcrText(hybridResult.text);
            hybridText = localCanonicalizeOcrText(hybridText, config);
            hybridScore = localOcrCandidateScore(hybridText, hybridResult.confidence, hybridResult.success, config) + 0.10;
            if hybridScore > bestScore
                bestScore = hybridScore;
                bestResult = hybridResult;
                bestResult.text = hybridText;
                bestVariantName = "hybrid_two_row";
                bestVariantImage = localPreferredOcrPreview(plateImage, hints);
            end
        end
    end

    [bestResult, bestVariantName, bestVariantImage] = localResolveWeakSuffixCandidate( ...
        bestResult, bestVariantName, bestVariantImage, attemptedResults, bestScore, config);
    [bestResult, bestVariantName, bestVariantImage] = localResolveSupportedAttemptCandidate( ...
        bestResult, bestVariantName, bestVariantImage, attemptedResults, config, hints);

    [repairedText, repairMeta] = repairStructuredPlateText(bestResult.text, config, struct( ...
        "confidence", double(bestResult.confidence), ...
        "layoutHint", localHintLayout(hints), ...
        "attemptedTexts", localAttemptedTexts(attemptedResults)));
    if repairMeta.changed
        bestResult.text = repairedText;
        if strlength(bestVariantName) > 0
            bestVariantName = bestVariantName + "+" + repairMeta.label;
        else
            bestVariantName = repairMeta.label;
        end
    end

    [refinedText, refinementLabel] = localRefineRecognizedText(bestResult.text, bestResult.confidence, config);
    if strlength(refinementLabel) > 0
        bestResult.text = refinedText;
        if strlength(bestVariantName) > 0
            bestVariantName = bestVariantName + "+" + refinementLabel;
        else
            bestVariantName = refinementLabel;
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
        "preparedPlateImage", bestVariantImage, ...
        "layoutHint", localHintLayout(hints), ...
        "attemptedResults", {attemptedResults}, ...
        "structuredRepair", repairMeta, ...
        "matlabOcr", bestResult);
end

function variants = localOcrVariants(plateImage, config, hints)
    grayImage = localGrayPlate(plateImage);
    preparedImage = prepareTesseractPlateImage(plateImage, config);
    invgrayImage = imcomplement(grayImage);
    binaryImage = localAdaptiveBinary(grayImage, "bright");
    darkBinaryImage = localAdaptiveBinary(grayImage, "dark");

    variants = struct("name", {}, "layout", {}, "image", {}, "source", {});
    variants = localAppendVariant(variants, "raw_line", "Line", plateImage, "full_plate");
    variants = localAppendVariant(variants, "gray_line", "Line", grayImage, "full_plate");
    variants = localAppendVariant(variants, "binary_line", "Line", binaryImage, "full_plate");
    variants = localAppendVariant(variants, "darkbinary_line", "Line", darkBinaryImage, "full_plate");
    variants = localAppendVariant(variants, "invgray_line", "Line", invgrayImage, "full_plate");
    variants = localAppendVariant(variants, "prepared_line", "Line", preparedImage, "full_plate");
    variants = localAppendVariant(variants, "raw_block", "Block", plateImage, "full_plate");
    variants = localAppendVariant(variants, "gray_block", "Block", grayImage, "full_plate");
    variants = localAppendVariant(variants, "invgray_block", "Block", invgrayImage, "full_plate");
    variants = localAppendVariant(variants, "prepared_block", "Block", preparedImage, "full_plate");
    variants = localAppendVariant(variants, "prepared_word", "Word", preparedImage, "full_plate");

    if isstruct(hints) && isfield(hints, "textOnlyPlate") && ~isempty(hints.textOnlyPlate)
        textOnlyPlate = hints.textOnlyPlate;
        textOnlyGray = localGrayPlate(textOnlyPlate);
        textOnlyPrepared = prepareTesseractPlateImage(textOnlyPlate, config);
        textOnlyBinary = localAdaptiveBinary(textOnlyGray, "bright");
        textOnlyDarkBinary = localAdaptiveBinary(textOnlyGray, "dark");
        textOnlyInv = imcomplement(textOnlyGray);
        textOnlyLeftTrimGray = localTrimPlateColumns(textOnlyGray, ...
            double(config.rectification.textVariantLeftTrimRatio), 0);
        textOnlyLeftTrimPrepared = prepareTesseractPlateImage(textOnlyLeftTrimGray, config);
        textOnlyRightTrimGray = localTrimPlateColumns(textOnlyGray, ...
            0, double(config.rectification.textVariantRightTrimRatio));
        textOnlyRightTrimPrepared = prepareTesseractPlateImage(textOnlyRightTrimGray, config);

        variants = localAppendVariant(variants, "textonly_line", "Line", textOnlyGray, "text_only");
        variants = localAppendVariant(variants, "textonly_binary_line", "Line", textOnlyBinary, "text_only");
        variants = localAppendVariant(variants, "textonly_darkbinary_line", "Line", textOnlyDarkBinary, "text_only");
        variants = localAppendVariant(variants, "textonly_prepared_line", "Line", textOnlyPrepared, "text_only");
        variants = localAppendVariant(variants, "textonly_block", "Block", textOnlyGray, "text_only");
        variants = localAppendVariant(variants, "textonly_inv_block", "Block", textOnlyInv, "text_only");
        variants = localAppendVariant(variants, "textonly_prepared_block", "Block", textOnlyPrepared, "text_only");
        variants = localAppendVariant(variants, "textonly_lefttrim_line", "Line", textOnlyLeftTrimGray, "text_only_trimmed");
        variants = localAppendVariant(variants, "textonly_lefttrim_block", "Block", textOnlyLeftTrimPrepared, "text_only_trimmed");
        variants = localAppendVariant(variants, "textonly_righttrim_line", "Line", textOnlyRightTrimGray, "text_only_trimmed");
        variants = localAppendVariant(variants, "textonly_righttrim_block", "Block", textOnlyRightTrimPrepared, "text_only_trimmed");
    end

    if isstruct(hints) && isfield(hints, "rowCompositePlate") && ~isempty(hints.rowCompositePlate)
        rowCompositePlate = hints.rowCompositePlate;
        rowCompositeGray = localGrayPlate(rowCompositePlate);
        rowCompositePrepared = prepareTesseractPlateImage(rowCompositePlate, config);
        rowCompositeBinary = localAdaptiveBinary(rowCompositeGray, "dark");
        rowCompositeInv = imcomplement(rowCompositeGray);

        variants = localAppendVariant(variants, "rowcomposite_line", "Line", rowCompositeGray, "row_composite");
        variants = localAppendVariant(variants, "rowcomposite_binary_line", "Line", rowCompositeBinary, "row_composite");
        variants = localAppendVariant(variants, "rowcomposite_prepared_line", "Line", rowCompositePrepared, "row_composite");
        variants = localAppendVariant(variants, "rowcomposite_block", "Block", rowCompositeGray, "row_composite");
        variants = localAppendVariant(variants, "rowcomposite_inv_block", "Block", rowCompositeInv, "row_composite");
        variants = localAppendVariant(variants, "rowcomposite_prepared_block", "Block", rowCompositePrepared, "row_composite");
        variants = localAppendVariant(variants, "rowcomposite_word", "Word", rowCompositePrepared, "row_composite");
    end
end

function [ocrResult, attemptRecord] = localRowwiseOcrCandidate(config, hints)
    ocrResult = struct("text", "", "confidence", 0, "success", false, "errorMessage", "");
    attemptRecord = [];

    if ~isstruct(hints) || ~isfield(hints, "rowImages") || numel(hints.rowImages) < 2
        return;
    end

    rowImages = hints.rowImages;
    rowTexts = strings(numel(rowImages), 1);
    rowConfidences = zeros(numel(rowImages), 1);
    rowAttempts = cell(numel(rowImages), 1);

    for i = 1:numel(rowImages)
        if isempty(rowImages{i})
            return;
        end

        [chosenResult, chosenImage, chosenName] = localBestRowResult(rowImages{i}, i, config, hints);

        rowTexts(i) = localCanonicalizeOcrText(localNormalizeOcrText(chosenResult.text), config);
        rowConfidences(i) = max(0, min(1, double(chosenResult.confidence)));
        rowAttempts{i} = struct( ...
            "rowIndex", i, ...
            "name", chosenName, ...
            "image", chosenImage, ...
            "result", chosenResult);
    end

    combinedText = join(rowTexts, "");
    combinedText = localCanonicalizeOcrText(localNormalizeOcrText(combinedText), config);
    successMask = strlength(rowTexts) > 0;
    if ~any(successMask)
        return;
    end

    ocrResult.text = combinedText;
    ocrResult.confidence = mean(rowConfidences(successMask));
    ocrResult.success = strlength(combinedText) > 0;
    ocrResult.errorMessage = "";
    ocrResult.rowTexts = rowTexts;
    attemptRecord = struct( ...
        "name", "rowwise_combo", ...
        "layout", "Line", ...
        "source", "rowwise_combo", ...
        "image", localPreferredOcrPreview([], hints), ...
        "result", ocrResult, ...
        "rows", {rowAttempts});
end

function [hybridResult, attemptRecord] = localHybridTwoRowCandidate(bestResult, rowwiseResult, config, hints)
    hybridResult = struct("text", "", "confidence", 0, "success", false, "errorMessage", "");
    attemptRecord = [];

    if localHintLayout(hints) ~= "two_row"
        return;
    end
    if ~isfield(rowwiseResult, "rowTexts") || numel(rowwiseResult.rowTexts) < 2
        return;
    end

    topLetters = regexp(char(localNormalizeOcrText(bestResult.text)), "^[A-Z]{1,4}", "match", "once");
    bottomDigits = regexp(char(localNormalizeOcrText(rowwiseResult.rowTexts(2))), "[0-9]{1,4}", "match", "once");
    if isempty(topLetters) || isempty(bottomDigits)
        return;
    end

    hybridText = string(topLetters) + string(bottomDigits);
    hybridResult.text = hybridText;
    hybridResult.confidence = mean([double(bestResult.confidence) double(rowwiseResult.confidence)]);
    hybridResult.success = strlength(hybridText) > 0;
    hybridResult.errorMessage = "";

    attemptRecord = struct( ...
        "name", "hybrid_two_row", ...
        "layout", "Line", ...
        "source", "hybrid_two_row", ...
        "image", localPreferredOcrPreview([], hints), ...
        "result", hybridResult);
end

function [chosenResult, chosenImage, chosenName] = localBestRowResult(rowImage, rowIndex, config, hints)
    grayRowImage = localGrayPlate(rowImage);
    preparedRowImage = prepareTesseractPlateImage(rowImage, config);
    binaryDarkRowImage = localAdaptiveBinary(grayRowImage, "dark");
    binaryBrightRowImage = localAdaptiveBinary(grayRowImage, "bright");
    invGrayRowImage = imcomplement(grayRowImage);

    candidateSpecs = { ...
        struct("name", "gray_line", "image", grayRowImage, "layoutAnalysis", "line", "textLayout", "Line", "characterSet", config.classification.matlabOcrCharacterSet); ...
        struct("name", "gray_word", "image", grayRowImage, "layoutAnalysis", "word", "textLayout", "Word", "characterSet", config.classification.matlabOcrCharacterSet); ...
        struct("name", "prepared_line", "image", preparedRowImage, "layoutAnalysis", "line", "textLayout", "Line", "characterSet", config.classification.matlabOcrCharacterSet); ...
        struct("name", "prepared_word", "image", preparedRowImage, "layoutAnalysis", "word", "textLayout", "Word", "characterSet", config.classification.matlabOcrCharacterSet); ...
        struct("name", "binary_dark_line", "image", binaryDarkRowImage, "layoutAnalysis", "line", "textLayout", "Line", "characterSet", config.classification.matlabOcrCharacterSet); ...
        struct("name", "binary_bright_line", "image", binaryBrightRowImage, "layoutAnalysis", "line", "textLayout", "Line", "characterSet", config.classification.matlabOcrCharacterSet); ...
        struct("name", "invgray_line", "image", invGrayRowImage, "layoutAnalysis", "line", "textLayout", "Line", "characterSet", config.classification.matlabOcrCharacterSet) ...
    };

    if localHintLayout(hints) == "two_row"
        if rowIndex == 1
            candidateSpecs{end + 1} = struct( ...
                "name", "letters_line", ...
                "image", preparedRowImage, ...
                "layoutAnalysis", "line", ...
                "textLayout", "Line", ...
                "characterSet", "ABCDEFGHIJKLMNOPQRSTUVWXYZ");
            candidateSpecs{end + 1} = struct( ...
                "name", "letters_word", ...
                "image", preparedRowImage, ...
                "layoutAnalysis", "word", ...
                "textLayout", "Word", ...
                "characterSet", "ABCDEFGHIJKLMNOPQRSTUVWXYZ");
            candidateSpecs{end + 1} = struct( ...
                "name", "letters_binary_line", ...
                "image", binaryDarkRowImage, ...
                "layoutAnalysis", "line", ...
                "textLayout", "Line", ...
                "characterSet", "ABCDEFGHIJKLMNOPQRSTUVWXYZ");
        elseif rowIndex == 2
            candidateSpecs{end + 1} = struct( ...
                "name", "digits_word", ...
                "image", preparedRowImage, ...
                "layoutAnalysis", "word", ...
                "textLayout", "Word", ...
                "characterSet", "0123456789");
            candidateSpecs{end + 1} = struct( ...
                "name", "digits_line", ...
                "image", grayRowImage, ...
                "layoutAnalysis", "line", ...
                "textLayout", "Line", ...
                "characterSet", "0123456789");
            candidateSpecs{end + 1} = struct( ...
                "name", "digits_prepared_line", ...
                "image", preparedRowImage, ...
                "layoutAnalysis", "line", ...
                "textLayout", "Line", ...
                "characterSet", "0123456789");
            candidateSpecs{end + 1} = struct( ...
                "name", "digits_binary_word", ...
                "image", binaryDarkRowImage, ...
                "layoutAnalysis", "word", ...
                "textLayout", "Word", ...
                "characterSet", "0123456789");
            candidateSpecs{end + 1} = struct( ...
                "name", "digits_binary_line", ...
                "image", binaryDarkRowImage, ...
                "layoutAnalysis", "line", ...
                "textLayout", "Line", ...
                "characterSet", "0123456789");
        end
    end

    bestScore = -inf;
    chosenResult = struct("text", "", "confidence", 0, "success", false, "errorMessage", "");
    chosenImage = grayRowImage;
    chosenName = "gray_line";

    for i = 1:numel(candidateSpecs)
        spec = candidateSpecs{i};
        rowConfig = config;
        rowConfig.classification.matlabOcrLayoutAnalysis = string(spec.layoutAnalysis);
        rowConfig.classification.matlabOcrTextLayout = string(spec.textLayout);
        rowConfig.classification.matlabOcrCharacterSet = string(spec.characterSet);
        rowResult = runMatlabOCR(spec.image, rowConfig);
        rowText = localNormalizeRowTextByExpectation(rowResult.text, rowIndex, hints);
        rowResult.text = rowText;
        rowResult.success = strlength(rowText) > 0;
        rowScore = localRowResultScore(rowText, rowResult, rowIndex, hints) + ...
            localRowVariantPreference(spec.name, rowIndex, hints);
        if rowScore > bestScore
            bestScore = rowScore;
            chosenResult = rowResult;
            chosenImage = spec.image;
            chosenName = string(spec.name);
        end
    end
end

function normalizedText = localNormalizeRowTextByExpectation(rowText, rowIndex, hints)
    normalizedText = localNormalizeOcrText(rowText);
    if localHintLayout(hints) ~= "two_row" || strlength(normalizedText) == 0
        return;
    end

    chars = char(normalizedText);
    for i = 1:numel(chars)
        if rowIndex == 1
            chars(i) = localAlphaRegionChar(chars(i));
        else
            chars(i) = localDigitRegionChar(chars(i));
        end
    end

    normalizedText = string(chars);
    if rowIndex == 1
        normalizedText = regexprep(normalizedText, "[^A-Z]", "");
    else
        normalizedText = regexprep(normalizedText, "[^0-9]", "");
    end
end

function bonus = localRowVariantPreference(specName, rowIndex, hints)
    bonus = 0;
    if localHintLayout(hints) ~= "two_row"
        return;
    end

    specName = string(specName);
    if rowIndex == 1
        if contains(specName, "letters")
            bonus = bonus + 0.08;
        end
    else
        if contains(specName, "digits")
            bonus = bonus + 0.10;
        end
    end

    if contains(specName, "prepared")
        bonus = bonus + 0.03;
    elseif contains(specName, "binary")
        bonus = bonus + 0.02;
    end
end

function score = localRowResultScore(rowText, rowResult, rowIndex, hints)
    rowText = string(rowText);
    score = double(rowResult.confidence);
    if ~logical(rowResult.success) || strlength(rowText) == 0
        score = -1;
        return;
    end

    letterCount = strlength(regexprep(rowText, "[^A-Z]", ""));
    digitCount = strlength(regexprep(rowText, "[^0-9]", ""));
    totalCount = strlength(rowText);

    if localHintLayout(hints) == "two_row"
        if rowIndex == 1
            score = score + 0.35 * double(letterCount == totalCount) + ...
                0.16 * double(totalCount >= 1 && totalCount <= 4) - ...
                0.18 * double(totalCount > 4) - ...
                0.20 * double(digitCount > 0);
        elseif rowIndex == 2
            score = score + 0.40 * double(digitCount == totalCount) + ...
                0.24 * double(totalCount >= 1 && totalCount <= 4) - ...
                0.20 * double(totalCount > 4) - ...
                0.24 * double(letterCount > 0);
        end
    end
end

function bonus = localRowwisePreference(hints)
    bonus = 0;
    if localHintLayout(hints) == "two_row"
        bonus = 0.24;
    end
end

function variants = localAppendVariant(variants, name, layout, imageValue, source)
    if isempty(imageValue)
        return;
    end

    variants(end + 1) = struct( ... %#ok<AGROW>
        "name", string(name), ...
        "layout", string(layout), ...
        "image", imageValue, ...
        "source", string(source));
end

function bonus = localVariantPreference(variant, hints)
    bonus = 0;
    layoutHint = localHintLayout(hints);
    source = string(variant.source);
    layout = string(variant.layout);

    if layoutHint == "two_row"
        if source == "row_composite" && layout == "Line"
            bonus = bonus + 0.18;
        elseif source == "text_only" && layout == "Block"
            bonus = bonus + 0.12;
        elseif source == "text_only"
            bonus = bonus + 0.05;
        elseif source == "full_plate" && layout == "Line"
            bonus = bonus - 0.03;
        end
    else
        if source == "text_only" && layout == "Line"
            bonus = bonus + 0.12;
        elseif source == "text_only" && layout == "Block"
            bonus = bonus + 0.07;
        elseif source == "text_only_trimmed" && layout == "Line"
            bonus = bonus + 0.11;
        elseif source == "text_only_trimmed" && layout == "Block"
            bonus = bonus + 0.09;
        elseif source == "full_plate" && layout == "Line"
            bonus = bonus + 0.03;
        end
    end

    if isstruct(hints) && isfield(hints, "profileName") && string(hints.profileName) == "two_row" && ...
            source == "row_composite"
        bonus = bonus + 0.05;
    end
end

function trimmedImage = localTrimPlateColumns(imageValue, leftTrimRatio, rightTrimRatio)
    trimmedImage = imageValue;
    if isempty(imageValue)
        return;
    end

    imageWidth = size(imageValue, 2);
    if imageWidth < 12
        return;
    end

    leftTrim = max(0, round(imageWidth * max(0, leftTrimRatio)));
    rightTrim = max(0, round(imageWidth * max(0, rightTrimRatio)));
    startColumn = min(imageWidth, 1 + leftTrim);
    endColumn = max(startColumn + 3, imageWidth - rightTrim);
    endColumn = min(imageWidth, endColumn);
    if endColumn <= startColumn
        return;
    end

    trimmedImage = imageValue(:, startColumn:endColumn, :);
end

function grayImage = localGrayPlate(plateImage)
    grayImage = plateImage;
    if ndims(grayImage) == 3
        grayImage = im2gray(grayImage);
    end
end

function binaryImage = localAdaptiveBinary(grayImage, polarity)
    binaryImage = imbinarize(im2single(grayImage), "adaptive", ...
        "ForegroundPolarity", polarity, ...
        "Sensitivity", 0.42);
    binaryImage = uint8(binaryImage) * 255;
end

function previewImage = localPreferredOcrPreview(defaultImage, hints)
    previewImage = defaultImage;
    if isstruct(hints) && isfield(hints, "rowCompositePlate") && ~isempty(hints.rowCompositePlate)
        previewImage = hints.rowCompositePlate;
    elseif isstruct(hints) && isfield(hints, "textOnlyPlate") && ~isempty(hints.textOnlyPlate)
        previewImage = hints.textOnlyPlate;
    elseif isstruct(hints) && isfield(hints, "rowImages") && numel(hints.rowImages) >= 1 && ~isempty(hints.rowImages{1})
        previewImage = hints.rowImages{1};
    end
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
    if localHasLeadingZeroDigitBlock(normalized)
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

function metadata = localEmptyMetadata(plateImage, hints)
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
        "preparedPlateImage", plateImage, ...
        "layoutHint", localHintLayout(hints), ...
        "attemptedResults", {cell(0, 1)}, ...
        "structuredRepair", struct( ...
            "changed", false, ...
            "label", "", ...
            "originalText", "", ...
            "selectedText", "", ...
            "originalScore", 0, ...
            "selectedScore", 0, ...
            "editCount", 0), ...
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

function outputChar = localAlphaRegionChar(inputChar)
    outputChar = upper(char(inputChar));
    switch outputChar
        case '0'
            outputChar = 'O';
        case '1'
            outputChar = 'I';
        case '2'
            outputChar = 'Z';
        case '5'
            outputChar = 'S';
        case '6'
            outputChar = 'G';
        case '7'
            outputChar = 'T';
        case '8'
            outputChar = 'B';
    end
end

function outputChar = localDigitRegionChar(inputChar)
    outputChar = upper(char(inputChar));
    switch outputChar
        case {'O', 'Q', 'D', 'U'}
            outputChar = '0';
        case {'I', 'L', 'T'}
            outputChar = '1';
        case 'Z'
            outputChar = '2';
        case 'S'
            outputChar = '5';
        case 'G'
            outputChar = '6';
        case 'B'
            outputChar = '8';
    end
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

function layoutHint = localHintLayout(hints)
    layoutHint = "unknown";
    if isstruct(hints) && isfield(hints, "layoutHint") && strlength(string(hints.layoutHint)) > 0
        layoutHint = string(hints.layoutHint);
    elseif isstruct(hints) && isfield(hints, "profileName") && string(hints.profileName) == "two_row"
        layoutHint = "two_row";
    end
end

function [bestResult, bestVariantName, bestVariantImage] = localResolveWeakSuffixCandidate( ...
        bestResult, bestVariantName, bestVariantImage, attemptedResults, bestScore, config)
    bestResult.text = localCanonicalizeOcrText(localNormalizeOcrText(bestResult.text), config);
    normalized = string(bestResult.text);
    if strlength(normalized) == 0
        return;
    end

    allowSuffixLetter = isfield(config, "classification") && ...
        isfield(config.classification, "allowSuffixLetter") && ...
        logical(config.classification.allowSuffixLetter);
    if ~allowSuffixLetter
        return;
    end

    [~, ~, suffixLength, isOrdered] = localOcrTextShape(normalized);
    if suffixLength ~= 1 || ~isOrdered || double(bestResult.confidence) > 0.35
        return;
    end

    trimmedText = extractBefore(normalized, strlength(normalized));
    if isempty(regexp(char(trimmedText), "^[A-Z]{1,4}[0-9]{1,4}$", "once"))
        return;
    end

    supportCount = 0;
    bestAlternativeScore = -inf;
    bestAlternativeResult = struct([]);
    bestAlternativeName = "";
    bestAlternativeImage = [];

    for i = 1:numel(attemptedResults)
        attempt = attemptedResults{i};
        if ~isstruct(attempt) || ~isfield(attempt, "result")
            continue;
        end

        alternativeText = localCanonicalizeOcrText(localNormalizeOcrText(attempt.result.text), config);
        if alternativeText ~= trimmedText
            continue;
        end

        supportCount = supportCount + 1;
        alternativeScore = localOcrCandidateScore( ...
            alternativeText, attempt.result.confidence, attempt.result.success, config);
        if alternativeScore > bestAlternativeScore
            bestAlternativeScore = alternativeScore;
            bestAlternativeResult = attempt.result;
            bestAlternativeResult.text = alternativeText;
            if isfield(attempt, "name")
                bestAlternativeName = string(attempt.name);
            end
            if isfield(attempt, "image")
                bestAlternativeImage = attempt.image;
            end
        end
    end

    if supportCount < 2 || bestAlternativeScore < (bestScore - 0.12) || isempty(bestAlternativeResult)
        return;
    end

    bestResult = bestAlternativeResult;
    bestResult.text = trimmedText;
    if strlength(bestAlternativeName) > 0
        bestVariantName = bestAlternativeName + "+trim_weak_suffix";
    elseif strlength(bestVariantName) > 0
        bestVariantName = bestVariantName + "+trim_weak_suffix";
    else
        bestVariantName = "trim_weak_suffix";
    end
    if ~isempty(bestAlternativeImage)
        bestVariantImage = bestAlternativeImage;
    end
end

function attemptedTexts = localAttemptedTexts(attemptedResults)
    attemptedTexts = strings(0, 1);
    for i = 1:numel(attemptedResults)
        attempt = attemptedResults{i};
        if ~isstruct(attempt) || ~isfield(attempt, "result") || ~isstruct(attempt.result)
            continue;
        end

        candidateText = localNormalizeOcrText(attempt.result.text);
        if strlength(candidateText) == 0
            continue;
        end
        attemptedTexts(end + 1, 1) = candidateText; %#ok<AGROW>
    end
end

function [bestResult, bestVariantName, bestVariantImage] = localResolveSupportedAttemptCandidate( ...
        bestResult, bestVariantName, bestVariantImage, attemptedResults, config, hints)
    normalizedBest = localCanonicalizeOcrText(localNormalizeOcrText(bestResult.text), config);
    if strlength(normalizedBest) == 0 || numel(attemptedResults) == 0
        return;
    end

    if localHintLayout(hints) ~= "two_row"
        return;
    end

    [bestPrefixLength, bestDigitLength, bestSuffixLength, bestOrdered] = localOcrTextShape(normalizedBest);
    if ~bestOrdered || bestDigitLength < 3 || bestPrefixLength < 2 || bestSuffixLength > 1
        return;
    end

    bestDigitBlock = extractBetween(normalizedBest, bestPrefixLength + 1, bestPrefixLength + bestDigitLength);
    bestSupport = localAttemptTextSupport(normalizedBest, attemptedResults, config);
    bestStatePriority = localStatePriority(identifyState(normalizedBest, config.malaysiaRules));

    preferredText = normalizedBest;
    preferredSupport = bestSupport;
    preferredScore = -inf;
    preferredResult = struct([]);
    preferredName = "";
    preferredImage = [];

    for i = 1:numel(attemptedResults)
        attempt = attemptedResults{i};
        if ~isstruct(attempt) || ~isfield(attempt, "result") || ~isstruct(attempt.result)
            continue;
        end

        candidateText = localCanonicalizeOcrText(localNormalizeOcrText(attempt.result.text), config);
        if strlength(candidateText) == 0 || candidateText == normalizedBest
            continue;
        end

        [candidatePrefixLength, candidateDigitLength, candidateSuffixLength, candidateOrdered] = localOcrTextShape(candidateText);
        if ~candidateOrdered || candidateSuffixLength > 1 || ...
                candidatePrefixLength ~= bestPrefixLength || candidateDigitLength ~= bestDigitLength
            continue;
        end

        candidateDigitBlock = extractBetween(candidateText, candidatePrefixLength + 1, candidatePrefixLength + candidateDigitLength);
        if candidateDigitBlock ~= bestDigitBlock
            continue;
        end

        if localEditCount(normalizedBest, candidateText) ~= 1
            continue;
        end

        if isempty(regexp(char(candidateText), "^[A-Z]{1,4}[0-9]{3,4}[A-Z]?$", "once"))
            continue;
        end

        candidatePriority = localStatePriority(identifyState(candidateText, config.malaysiaRules));
        if candidatePriority + 5 < bestStatePriority
            continue;
        end

        candidateSupport = localAttemptTextSupport(candidateText, attemptedResults, config);
        if candidateSupport < 1.5 || candidateSupport <= (bestSupport + 0.45)
            continue;
        end

        candidateScore = candidateSupport + 0.10 * min(1, candidatePriority / 100);
        if candidateScore > preferredScore
            preferredText = candidateText;
            preferredSupport = candidateSupport;
            preferredScore = candidateScore;
            preferredResult = attempt.result;
            preferredResult.text = candidateText;
            if isfield(attempt, "name")
                preferredName = string(attempt.name);
            end
            if isfield(attempt, "image")
                preferredImage = attempt.image;
            end
        end
    end

    if preferredText == normalizedBest || preferredSupport <= bestSupport || isempty(preferredResult)
        return;
    end

    bestResult = preferredResult;
    bestResult.text = preferredText;
    if strlength(preferredName) > 0
        bestVariantName = preferredName + "+attempt_support";
    elseif strlength(bestVariantName) > 0
        bestVariantName = bestVariantName + "+attempt_support";
    else
        bestVariantName = "attempt_support";
    end
    if ~isempty(preferredImage)
        bestVariantImage = preferredImage;
    end
end

function supportScore = localAttemptTextSupport(candidateText, attemptedResults, config)
    supportScore = 0;
    normalizedTarget = localCanonicalizeOcrText(localNormalizeOcrText(candidateText), config);
    for i = 1:numel(attemptedResults)
        attempt = attemptedResults{i};
        if ~isstruct(attempt) || ~isfield(attempt, "result") || ~isstruct(attempt.result)
            continue;
        end

        attemptText = localCanonicalizeOcrText(localNormalizeOcrText(attempt.result.text), config);
        if attemptText ~= normalizedTarget
            continue;
        end

        supportScore = supportScore + localAttemptSupportWeight(attempt);
    end
end

function weight = localAttemptSupportWeight(attempt)
    weight = 0.35;
    sourceValue = "";
    nameValue = "";
    if isfield(attempt, "source")
        sourceValue = string(attempt.source);
    end
    if isfield(attempt, "name")
        nameValue = string(attempt.name);
    end

    switch sourceValue
        case "text_only"
            weight = 1.00;
        case "text_only_trimmed"
            weight = 0.75;
        case "full_plate"
            weight = 0.70;
        case "row_composite"
            weight = 0.30;
        case {"rowwise_combo", "hybrid_two_row"}
            weight = 0.25;
    end

    if startsWith(nameValue, "textonly_")
        weight = max(weight, 0.85);
    elseif startsWith(nameValue, "rowcomposite_")
        weight = min(weight, 0.30);
    elseif nameValue == "rowwise_combo" || nameValue == "hybrid_two_row"
        weight = min(weight, 0.25);
    end
end

function [refinedText, refinementLabel] = localRefineRecognizedText(textValue, confidenceValue, config)
    refinedText = upper(regexprep(string(textValue), "[^A-Z0-9]", ""));
    refinementLabel = "";
    if strlength(refinedText) == 0
        return;
    end

    [prefixLength, digitLength, suffixLength, ~] = localOcrTextShape(refinedText);
    originalState = identifyState(refinedText, config.malaysiaRules);
    originalPriority = localStatePriority(originalState);
    bestScore = localRefinementScore(refinedText, originalPriority, config);

    if suffixLength > 1
        trailingTrim = extractBefore(refinedText, strlength(refinedText));
        trailingScore = localRefinementScore(trailingTrim, localStatePriority(identifyState(trailingTrim, config.malaysiaRules)), config);
        if trailingScore > bestScore + 0.06
            refinedText = trailingTrim;
            refinementLabel = "trim_tail";
            bestScore = trailingScore;
            [prefixLength, digitLength, suffixLength, ~] = localOcrTextShape(refinedText); %#ok<ASGLU>
            originalPriority = localStatePriority(identifyState(refinedText, config.malaysiaRules));
        end
    end

    allowSuffixLetter = isfield(config, "classification") && ...
        isfield(config.classification, "allowSuffixLetter") && ...
        logical(config.classification.allowSuffixLetter);
    if allowSuffixLetter && suffixLength == 1 && double(confidenceValue) <= 0.25
        trailingTrim = extractBefore(refinedText, strlength(refinedText));
        if ~isempty(regexp(char(trailingTrim), "^[A-Z]{1,4}[0-9]{1,4}$", "once"))
            trailingPriority = localStatePriority(identifyState(trailingTrim, config.malaysiaRules));
            trailingScore = localRefinementScore(trailingTrim, trailingPriority, config);
            if trailingScore >= bestScore - 0.02
                refinedText = trailingTrim;
                refinementLabel = "trim_weak_suffix";
                bestScore = trailingScore;
                [prefixLength, ~, ~, ~] = localOcrTextShape(refinedText);
                originalPriority = trailingPriority;
            end
        end
    end

    if prefixLength >= 2
        leadingTrim = extractAfter(refinedText, 1);
        trimmedState = identifyState(leadingTrim, config.malaysiaRules);
        trimmedPriority = localStatePriority(trimmedState);
        leadingScore = localRefinementScore(leadingTrim, trimmedPriority, config);
        if trimmedPriority > originalPriority
            if leadingScore > bestScore + 0.04
                refinedText = leadingTrim;
                refinementLabel = "trim_head";
            end
        elseif trimmedPriority == originalPriority && leadingScore > bestScore + 0.02
            refinedText = leadingTrim;
            refinementLabel = "trim_head";
        end
    end
end

function score = localRefinementScore(textValue, statePriority, config)
    normalized = upper(regexprep(string(textValue), "[^A-Z0-9]", ""));
    if strlength(normalized) == 0
        score = -inf;
        return;
    end

    [prefixLength, digitLength, suffixLength, isOrdered] = localOcrTextShape(normalized);
    totalLength = strlength(normalized);
    allowSuffixLetter = isfield(config, "classification") && ...
        isfield(config.classification, "allowSuffixLetter") && ...
        logical(config.classification.allowSuffixLetter);
    if allowSuffixLetter
        exactPattern = "^[A-Z]{1,4}[0-9]{1,4}[A-Z]?$";
    else
        exactPattern = "^[A-Z]{1,4}[0-9]{1,4}$";
    end

    regexScore = 0;
    if ~isempty(regexp(char(normalized), exactPattern, "once"))
        regexScore = 1.0;
    elseif ~isempty(regexp(char(normalized), "^[A-Z]{1,4}[0-9]{1,4}", "once"))
        regexScore = 0.55;
    end

    lengthScore = max(0, 1 - 0.18 * abs(totalLength - 7));
    structureScore = 0.30 * double(isOrdered) + ...
        0.25 * double(prefixLength >= 1 && prefixLength <= 4) + ...
        0.25 * double(digitLength >= 1 && digitLength <= 4) + ...
        0.20 * double(suffixLength <= 1);
    if localHasLeadingZeroDigitBlock(normalized)
        structureScore = max(0, structureScore - 0.35);
    end
    stateScore = min(1, statePriority / 100);
    score = 0.42 * regexScore + 0.24 * lengthScore + 0.20 * structureScore + 0.14 * stateScore;
end

function hasLeadingZero = localHasLeadingZeroDigitBlock(normalized)
    [prefixLength, digitLength, ~, isOrdered] = localOcrTextShape(normalized);
    hasLeadingZero = false;
    if ~isOrdered || digitLength < 2
        return;
    end

    digitBlock = extractBetween(normalized, prefixLength + 1, prefixLength + digitLength);
    if strlength(digitBlock) > 0 && startsWith(string(digitBlock), "0")
        hasLeadingZero = true;
    end
end

function priority = localStatePriority(stateInfo)
    priority = 0;
    if isstruct(stateInfo) && isfield(stateInfo, "matchedPriority")
        priority = double(stateInfo.matchedPriority);
    end
end

function editCount = localEditCount(sourceText, candidateText)
    sourceChars = char(string(sourceText));
    candidateChars = char(string(candidateText));
    comparedLength = min(numel(sourceChars), numel(candidateChars));
    editCount = sum(sourceChars(1:comparedLength) ~= candidateChars(1:comparedLength)) + ...
        abs(numel(sourceChars) - numel(candidateChars));
end
