function [selectedCandidateIndex, evaluatedCandidates] = rerankPlateCandidates(inputImage, detectorCandidates, config)
    %RERANKPLATECANDIDATES Evaluate top detector candidates using rectification and MATLAB OCR.

    config = validateConfig(config);
    evaluatedCandidates = localEmptyEvaluatedCandidates();

    if nargin < 2 || isempty(detectorCandidates)
        selectedCandidateIndex = [];
        return;
    end

    numCandidates = min(numel(detectorCandidates), config.reranking.maxCandidatesToEvaluate);
    imageSize = size(inputImage);

    for i = 1:numCandidates
        detectorCandidate = localNormalizeCandidate(detectorCandidates(i));
        paddedBBox = localPadBBox(detectorCandidate.bbox, imageSize, config.detection.platePadding, config);
        detectorPlate = localDetectorPlate(inputImage, detectorCandidate.bbox);

        [rectifiedPlate, rectifyMeta] = rectifyPlate(inputImage, paddedBBox, config, detectorCandidate);
        recognitionHints = localRecognitionHints(detectorCandidate, rectifyMeta);
        [recognizedText, recognitionMeta] = recognizeCharacters(rectifiedPlate, config, recognitionHints);

        stateLookupText = string(recognizedText);
        if isfield(recognitionMeta, "stateLookupText") && strlength(string(recognitionMeta.stateLookupText)) > 0
            stateLookupText = string(recognitionMeta.stateLookupText);
        end

        stateInfo = identifyState(stateLookupText, config.malaysiaRules);
        ocrInputPlate = localRecognitionPlate(recognitionMeta, rectifiedPlate);
        [scoreBreakdown, finalScore] = localScoreEvaluatedCandidate( ...
            detectorCandidate, ocrInputPlate, recognitionMeta, recognizedText, stateInfo, config);

        evaluatedCandidates(end + 1) = struct( ... %#ok<AGROW>
            "candidateIndex", i, ...
            "bbox", paddedBBox, ...
            "rawBBox", detectorCandidate.bbox, ...
            "branchName", detectorCandidate.branchName, ...
            "profileName", detectorCandidate.profileName, ...
            "layoutHint", localEvaluatedLayoutHint(detectorCandidate, rectifyMeta, recognitionMeta), ...
            "scale", detectorCandidate.scale, ...
            "detectorScore", detectorCandidate.score, ...
            "characterTextureScore", localGetCandidateField(detectorCandidate, "characterTextureScore"), ...
            "plateContrastScore", localGetCandidateField(detectorCandidate, "plateContrastScore"), ...
            "componentAlignmentScore", localGetCandidateField(detectorCandidate, "componentAlignmentScore"), ...
            "plateEvidenceScore", scoreBreakdown.plateEvidence, ...
            "recognizedText", string(recognizedText), ...
            "stateLookupText", stateLookupText, ...
            "stateInfo", stateInfo, ...
            "textLength", strlength(string(recognizedText)), ...
            "detectorPlate", detectorPlate, ...
            "rectifiedPlate", rectifiedPlate, ...
            "ocrInputPlate", ocrInputPlate, ...
            "rectifyMeta", rectifyMeta, ...
            "recognitionMeta", recognitionMeta, ...
            "recognitionPath", "matlab_ocr", ...
            "recognitionScore", scoreBreakdown.recognition, ...
            "regexScore", scoreBreakdown.regex, ...
            "stateScore", scoreBreakdown.state, ...
            "compositionScore", scoreBreakdown.composition, ...
            "lengthScore", scoreBreakdown.length, ...
            "structureScore", scoreBreakdown.structure, ...
            "framingScore", scoreBreakdown.framing, ...
            "emptyPenalty", scoreBreakdown.penalty, ...
            "scoreBreakdown", scoreBreakdown, ...
            "finalScore", finalScore, ...
            "selectionReason", localSelectionReason(scoreBreakdown, string(recognizedText), stateInfo));
    end

    if isempty(evaluatedCandidates)
        selectedCandidateIndex = [];
        return;
    end

    [~, sortIdx] = sort([evaluatedCandidates.finalScore], "descend");
    evaluatedCandidates = evaluatedCandidates(sortIdx);
    evaluatedCandidates = localApplyOverlapTextPreference(evaluatedCandidates, config);
    [~, sortIdx] = sort([evaluatedCandidates.finalScore], "descend");
    evaluatedCandidates = evaluatedCandidates(sortIdx);
    localShowEvaluatedCandidateDebug(evaluatedCandidates, config);
    selectedCandidateIndex = 1;
end

function detectorCandidate = localNormalizeCandidate(candidate)
    detectorCandidate = struct( ...
        "bbox", [], ...
        "score", 0, ...
        "branchName", "unknown", ...
        "profileName", "unknown", ...
        "layoutHint", "unknown", ...
        "scale", 1.0, ...
        "characterTextureScore", 0, ...
        "plateContrastScore", 0, ...
        "componentAlignmentScore", 0, ...
        "textComponentCount", 0, ...
        "rowCountEstimate", 0, ...
        "emptyRegionPenalty", 1);

    fields = fieldnames(candidate);
    for i = 1:numel(fields)
        detectorCandidate.(fields{i}) = candidate.(fields{i});
    end
end

function [scoreBreakdown, finalScore] = localScoreEvaluatedCandidate(detectorCandidate, ocrInputPlate, recognitionMeta, recognizedText, stateInfo, config)
    detectorScore = max(0, min(1, detectorCandidate.score));
    plateEvidenceScore = localDetectorEvidenceScore(detectorCandidate);
    recognitionScore = localRecognitionScore(recognitionMeta);
    [regexScore, regexLabel] = localRegexScore(string(recognizedText), config);
    stateScore = localStateScore(stateInfo, string(recognizedText));
    [compositionScore, compositionLabel] = localCompositionScore(string(recognizedText), config);
    [lengthScore, lengthLabel] = localLengthScore(string(recognizedText), config);
    [structureScore, structureLabel, prefixLength, digitLength, suffixLength] = ...
        localStructureScore(string(recognizedText), config);
    layoutHint = localEvaluatedLayoutHint(detectorCandidate, struct(), recognitionMeta);
    [framingScore, framingLabel, leftMargin, rightMargin] = ...
        localFramingScore(ocrInputPlate, string(recognizedText), layoutHint);
    [prefixScore, prefixLabel] = localPrefixPlausibilityScore(string(recognizedText), stateInfo, config);
    weakOcrFactor = localWeakOcrFactor(recognitionScore, config);
    ocrTrust = localOcrTrust(recognitionScore, prefixScore, weakOcrFactor);

    confidenceAdjustedRegexScore = min(1, ocrTrust * (0.85 * regexScore + 0.15 * recognitionScore));
    confidenceAdjustedStateScore = min(1, ocrTrust * (0.85 * stateScore + 0.15 * recognitionScore));
    confidenceAdjustedCompositionScore = min(1, ocrTrust * compositionScore);
    confidenceAdjustedLengthScore = min(1, ocrTrust * lengthScore);
    confidenceAdjustedStructureScore = min(1, ocrTrust * structureScore);

    weights = config.reranking.weights;
    baseScore = weights.detector * detectorScore + ...
        weights.plateEvidence * plateEvidenceScore + ...
        weights.recognition * recognitionScore + ...
        weights.regex * confidenceAdjustedRegexScore + ...
        weights.state * confidenceAdjustedStateScore + ...
        weights.composition * confidenceAdjustedCompositionScore + ...
        weights.length * confidenceAdjustedLengthScore + ...
        weights.structure * confidenceAdjustedStructureScore + ...
        weights.framing * framingScore;

    penalty = 0;
    if strlength(string(recognizedText)) == 0
        penalty = penalty + config.reranking.emptyCandidatePenalty;
    end
    if plateEvidenceScore < 0.25
        penalty = penalty + config.reranking.weakTexturePenalty;
    end
    if compositionScore < 0.25
        penalty = penalty + config.reranking.noDigitPenalty;
    elseif compositionScore < 0.55
        penalty = penalty + config.reranking.weakCompositionPenalty;
    end
    if strlength(regexprep(string(recognizedText), "[^A-Z]", "")) == 0 && strlength(string(recognizedText)) > 0
        penalty = penalty + config.reranking.noLetterPenalty;
    end
    if lengthScore < 0.35 && strlength(string(recognizedText)) > 0
        penalty = penalty + config.reranking.implausibleLengthPenalty;
    end
    if localHasLeadingZeroDigitBlock(string(recognizedText))
        penalty = penalty + config.reranking.leadingZeroDigitPenalty;
    end
    allowSuffixLetter = isfield(config.classification, "allowSuffixLetter") && ...
        logical(config.classification.allowSuffixLetter);
    if prefixLength <= 1 && digitLength >= 3 && leftMargin < 0.02
        penalty = penalty + config.reranking.truncatedCandidatePenalty;
    end
    if strlength(string(recognizedText)) > 0 && digitLength > 0 && digitLength <= 2
        penalty = penalty + config.reranking.shortReadPenalty;
    end
    if suffixLength > 0 && ~allowSuffixLetter
        penalty = penalty + config.reranking.forbiddenSuffixPenalty * min(suffixLength, 2);
    end
    if framingScore < 0.35 && strlength(string(recognizedText)) > 0
        penalty = penalty + config.reranking.weakFramingPenalty;
    end
    if strlength(string(recognizedText)) > 0 && prefixScore < 0.20
        penalty = penalty + config.reranking.invalidPrefixPenalty;
    end

    bonus = 0;
    switch regexLabel
        case "exact"
            bonus = bonus + config.reranking.regexExactBonus;
        case "partial"
            bonus = bonus + config.reranking.regexPartialBonus;
    end
    if compositionScore >= 0.85
        bonus = bonus + config.reranking.mixedAlphaDigitBonus;
    end
    if weakOcrFactor > 0
        bonus = bonus + weakOcrFactor * config.reranking.weakOcrDetectorBonus * detectorScore;
        bonus = bonus + weakOcrFactor * config.reranking.weakOcrPlateEvidenceBonus * plateEvidenceScore;
    end

    finalScore = max(0, min(1, baseScore + bonus - penalty));
    scoreBreakdown = struct( ...
        "detector", detectorScore, ...
        "plateEvidence", plateEvidenceScore, ...
        "characterTexture", localGetCandidateField(detectorCandidate, "characterTextureScore"), ...
        "plateContrast", localGetCandidateField(detectorCandidate, "plateContrastScore"), ...
        "componentAlignment", localGetCandidateField(detectorCandidate, "componentAlignmentScore"), ...
        "recognition", recognitionScore, ...
        "ocrTrust", ocrTrust, ...
        "weakOcrFactor", weakOcrFactor, ...
        "composition", compositionScore, ...
        "compositionLabel", string(compositionLabel), ...
        "length", lengthScore, ...
        "lengthLabel", string(lengthLabel), ...
        "structure", structureScore, ...
        "structureLabel", string(structureLabel), ...
        "prefix", prefixScore, ...
        "prefixLabel", string(prefixLabel), ...
        "prefixLength", prefixLength, ...
        "digitLength", digitLength, ...
        "suffixLength", suffixLength, ...
        "framing", framingScore, ...
        "framingLabel", string(framingLabel), ...
        "layoutHint", string(layoutHint), ...
        "leftMargin", leftMargin, ...
        "rightMargin", rightMargin, ...
        "regex", confidenceAdjustedRegexScore, ...
        "regexLabel", string(regexLabel), ...
        "state", confidenceAdjustedStateScore, ...
        "bonus", bonus, ...
        "penalty", penalty, ...
        "final", finalScore);
end

function evaluatedCandidates = localApplyOverlapTextPreference(evaluatedCandidates, config)
    if numel(evaluatedCandidates) < 2
        return;
    end

    overlapThreshold = double(config.reranking.substringOverlapThreshold);
    for i = 1:numel(evaluatedCandidates)
        textA = localNormalizedText(evaluatedCandidates(i).recognizedText);
        if strlength(textA) == 0
            continue;
        end

        for j = (i + 1):numel(evaluatedCandidates)
            textB = localNormalizedText(evaluatedCandidates(j).recognizedText);
            if strlength(textB) == 0 || textA == textB
                continue;
            end

            overlapScore = localBBoxOverlapScore(evaluatedCandidates(i).bbox, evaluatedCandidates(j).bbox);
            if overlapScore < overlapThreshold
                continue;
            end

            if contains(textA, textB) && strlength(textA) > strlength(textB)
                [evaluatedCandidates(i), evaluatedCandidates(j)] = localApplyPairwiseTextPreference( ...
                    evaluatedCandidates(i), evaluatedCandidates(j), textA, textB, config);
            elseif contains(textB, textA) && strlength(textB) > strlength(textA)
                [evaluatedCandidates(j), evaluatedCandidates(i)] = localApplyPairwiseTextPreference( ...
                    evaluatedCandidates(j), evaluatedCandidates(i), textB, textA, config);
            end
        end
    end
end

function [longCandidate, shortCandidate] = localApplyPairwiseTextPreference(longCandidate, shortCandidate, longText, shortText, config)
    [~, ~, longSuffixLength, longOrdered] = localTextShape(longText);
    [shortPrefixLength, shortDigitLength, shortSuffixLength, shortOrdered] = localTextShape(shortText);
    allowSuffixLetter = isfield(config.classification, "allowSuffixLetter") && ...
        logical(config.classification.allowSuffixLetter);

    if ~allowSuffixLetter && startsWith(longText, shortText) && longSuffixLength > 0 && ...
            shortSuffixLength == 0 && shortOrdered && longOrdered && ...
            shortPrefixLength >= 1 && shortPrefixLength <= 4 && ...
            shortDigitLength >= 1 && shortDigitLength <= 4
        longCandidate = localAdjustCandidateFinalScore( ...
            longCandidate, -config.reranking.overlapForbiddenSuffixPenalty, "overlap_forbidden_suffix_penalty");
        shortCandidate = localAdjustCandidateFinalScore( ...
            shortCandidate, config.reranking.overlapExactBaseBonus, "overlap_exact_base_bonus");
        return;
    end

    longCandidate = localAdjustCandidateFinalScore( ...
        longCandidate, config.reranking.overlapFullTextBonus, "overlap_fuller_text_bonus");
    shortCandidate = localAdjustCandidateFinalScore( ...
        shortCandidate, -config.reranking.overlapSubstringPenalty, "overlap_substring_penalty");
end

function candidate = localAdjustCandidateFinalScore(candidate, delta, reasonLabel)
    candidate.finalScore = max(0, min(1, candidate.finalScore + delta));
    if isfield(candidate, "scoreBreakdown") && isstruct(candidate.scoreBreakdown)
        candidate.scoreBreakdown.final = candidate.finalScore;
        if ~isfield(candidate.scoreBreakdown, "postAdjustments") || isempty(candidate.scoreBreakdown.postAdjustments)
            candidate.scoreBreakdown.postAdjustments = strings(0, 1);
        end
        candidate.scoreBreakdown.postAdjustments(end + 1, 1) = string(reasonLabel) + "=" + sprintf("%.2f", delta);
    end
    candidate.selectionReason = candidate.selectionReason + ", Adj=" + string(reasonLabel) + ...
        "(" + sprintf("%.2f", delta) + ")";
end

function overlapScore = localBBoxOverlapScore(boxA, boxB)
    iouScore = localIoU(boxA, boxB);
    areaA = max(boxA(3) * boxA(4), eps);
    areaB = max(boxB(3) * boxB(4), eps);

    x1 = max(boxA(1), boxB(1));
    y1 = max(boxA(2), boxB(2));
    x2 = min(boxA(1) + boxA(3), boxB(1) + boxB(3));
    y2 = min(boxA(2) + boxA(4), boxB(2) + boxB(4));
    intersectionArea = max(0, x2 - x1) * max(0, y2 - y1);
    containmentScore = intersectionArea / max(min(areaA, areaB), eps);
    overlapScore = max(iouScore, containmentScore);
end

function value = localIoU(boxA, boxB)
    x1 = max(boxA(1), boxB(1));
    y1 = max(boxA(2), boxB(2));
    x2 = min(boxA(1) + boxA(3), boxB(1) + boxB(3));
    y2 = min(boxA(2) + boxA(4), boxB(2) + boxB(4));

    intersectionArea = max(0, x2 - x1) * max(0, y2 - y1);
    areaA = boxA(3) * boxA(4);
    areaB = boxB(3) * boxB(4);
    unionArea = areaA + areaB - intersectionArea;

    if unionArea <= 0
        value = 0;
    else
        value = intersectionArea / unionArea;
    end
end

function normalized = localNormalizedText(textValue)
    normalized = upper(regexprep(string(textValue), "[^A-Z0-9]", ""));
end

function score = localRecognitionScore(recognitionMeta)
    score = 0;
    if isempty(recognitionMeta) || ~isstruct(recognitionMeta)
        return;
    end
    if isfield(recognitionMeta, "confidences") && ~isempty(recognitionMeta.confidences)
        score = mean(double(recognitionMeta.confidences));
    elseif isfield(recognitionMeta, "matlabOcr") && isstruct(recognitionMeta.matlabOcr) && ...
            isfield(recognitionMeta.matlabOcr, "confidence")
        score = double(recognitionMeta.matlabOcr.confidence);
    end
    score = max(0, min(1, score));
end

function weakOcrFactor = localWeakOcrFactor(recognitionScore, config)
    threshold = double(config.reranking.weakOcrRecognitionThreshold);
    threshold = max(threshold, eps);
    weakOcrFactor = max(0, min(1, (threshold - recognitionScore) / threshold));
end

function trust = localOcrTrust(recognitionScore, prefixScore, weakOcrFactor)
    baseTrust = max(0.18, min(1, 0.20 + 0.80 * recognitionScore));
    prefixTrust = 0.30 + 0.70 * max(0, min(1, prefixScore));
    weakOcrTrust = 1 - 0.55 * weakOcrFactor;
    trust = max(0.12, min(1, baseTrust * prefixTrust * weakOcrTrust));
end

function [score, label] = localRegexScore(recognizedText, config)
    normalized = upper(regexprep(string(recognizedText), "[^A-Z0-9]", ""));
    label = "none";

    if strlength(normalized) == 0
        score = 0;
        return;
    end

    allowSuffixLetter = isfield(config.classification, "allowSuffixLetter") && ...
        logical(config.classification.allowSuffixLetter);
    if allowSuffixLetter
        exactPattern = "^[A-Z]{1,4}[0-9]{1,4}[A-Z]?$";
    else
        exactPattern = "^[A-Z]{1,4}[0-9]{1,4}$";
    end
    partialPattern = "^[A-Z]{1,4}[0-9]{1,4}.*$";

    if ~isempty(regexp(normalized, exactPattern, "once"))
        score = 1.0;
        label = "exact";
    elseif ~isempty(regexp(normalized, partialPattern, "once"))
        score = 0.60;
        label = "partial";
    elseif any(startsWith(normalized, string({config.malaysiaRules.token})))
        score = 0.40;
        label = "prefix";
    else
        score = 0.05;
    end
end

function score = localStateScore(stateInfo, recognizedText)
    if isfield(stateInfo, "matched") && stateInfo.matched
        score = 1.0;
    elseif strlength(recognizedText) > 0
        score = 0.20;
    else
        score = 0;
    end
end

function [score, label] = localPrefixPlausibilityScore(recognizedText, stateInfo, config)
    normalized = upper(regexprep(string(recognizedText), "[^A-Z0-9]", ""));
    label = "empty";
    score = 0;

    if strlength(normalized) == 0
        return;
    end

    if isfield(stateInfo, "matched") && stateInfo.matched
        score = 1.0;
        label = "matched_rule";
        return;
    end

    prefixRules = config.malaysiaRules(strcmp(string({config.malaysiaRules.matcherType}), "prefix"));
    knownLeadTokens = unique(string({prefixRules.token}));
    leadingLetter = extractBefore(normalized, 2);

    if any(leadingLetter == knownLeadTokens)
        score = 0.62;
        label = "known_lead";
    else
        score = 0.05;
        label = "invalid_lead";
    end
end

function [score, label] = localCompositionScore(recognizedText, config)
    normalized = upper(regexprep(string(recognizedText), "[^A-Z0-9]", ""));
    label = "none";
    if strlength(normalized) == 0
        score = 0;
        return;
    end

    letterCount = strlength(regexprep(normalized, "[^A-Z]", ""));
    digitCount = strlength(regexprep(normalized, "[^0-9]", ""));
    totalCount = letterCount + digitCount;

    if letterCount >= 1 && letterCount <= 4 && digitCount >= 3 && digitCount <= 4 && totalCount >= 5 && totalCount <= 8
        score = 1.0;
        label = "balanced";
    elseif letterCount >= 1 && digitCount >= 1
        score = 0.55;
        label = "mixed";
    elseif digitCount == 0
        score = 0.15;
        label = "letters_only";
    elseif letterCount == 0
        score = 0.10;
        label = "digits_only";
    else
        score = 0.35;
        label = "weak";
    end

    if nargin >= 2 && isfield(config, "reranking") && isfield(config.reranking, "expectedCharacterRange")
        expectedRange = double(config.reranking.expectedCharacterRange);
        if totalCount < expectedRange(1) || totalCount > expectedRange(2)
            score = score * 0.75;
        end
    end
end

function [score, label] = localLengthScore(recognizedText, config)
    lengthValue = strlength(upper(regexprep(string(recognizedText), "[^A-Z0-9]", "")));
    expectedRange = double(config.reranking.expectedCharacterRange);

    if lengthValue == 0
        score = 0;
        label = "empty";
    else
        idealLength = 6.5;
        distance = abs(lengthValue - idealLength);
        if lengthValue >= expectedRange(1) && lengthValue <= expectedRange(2)
            score = max(0.35, 1.0 - 0.22 * distance);
            label = "expected";
        else
            score = max(0, 0.55 - 0.18 * distance);
            label = "out_of_range";
        end
    end
end

function [score, label, prefixLength, digitLength, suffixLength] = localStructureScore(recognizedText, config)
    normalized = upper(regexprep(string(recognizedText), "[^A-Z0-9]", ""));
    prefixLength = 0;
    digitLength = 0;
    suffixLength = 0;
    label = "empty";

    if strlength(normalized) == 0
        score = 0;
        return;
    end

    [prefixLength, digitLength, suffixLength, isOrdered] = localTextShape(normalized);
    totalLength = strlength(normalized);

    prefixScore = localPrefixStructureScore(prefixLength);
    digitScore = localDigitStructureScore(digitLength);
    if suffixLength == 0
        suffixScore = 1.0;
    elseif suffixLength == 1 && isfield(config.classification, "allowSuffixLetter") && ...
            logical(config.classification.allowSuffixLetter)
        suffixScore = 0.9;
    elseif suffixLength == 1
        suffixScore = 0.18;
    else
        suffixScore = 0.05;
    end

    orderScore = double(isOrdered);
    densityScore = max(0, min(1, totalLength / max(config.reranking.expectedCharacterRange(1), 1)));
    score = 0.42 * prefixScore + 0.28 * digitScore + 0.12 * suffixScore + ...
        0.10 * orderScore + 0.08 * densityScore;
    if localHasLeadingZeroDigitBlock(normalized)
        score = max(0, score - 0.35);
    end
    score = max(0, min(1, score));
    label = sprintf("P%d-D%d-S%d", prefixLength, digitLength, suffixLength);
end

function score = localPrefixStructureScore(prefixLength)
    switch prefixLength
        case 0
            score = 0;
        case 1
            score = 0.42;
        case 2
            score = 0.75;
        case 3
            score = 1.00;
        case 4
            score = 0.88;
        otherwise
            score = max(0.15, 0.80 - 0.12 * abs(prefixLength - 3));
    end
end

function score = localDigitStructureScore(digitLength)
    switch digitLength
        case 0
            score = 0;
        case 1
            score = 0.20;
        case 2
            score = 0.35;
        case 3
            score = 0.82;
        case 4
            score = 1.00;
        otherwise
            score = max(0.15, 0.85 - 0.15 * abs(digitLength - 4));
    end
end

function [prefixLength, digitLength, suffixLength, isOrdered] = localTextShape(normalized)
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
    orderedLength = prefixLength + digitLength + suffixLength;
    isOrdered = orderedLength == strlength(normalized) && ...
        suffixLength == strlength(suffix);
end

function hasLeadingZero = localHasLeadingZeroDigitBlock(textValue)
    normalized = upper(regexprep(string(textValue), "[^A-Z0-9]", ""));
    [prefixLength, digitLength, ~, isOrdered] = localTextShape(normalized);
    hasLeadingZero = false;
    if ~isOrdered || digitLength < 2
        return;
    end

    digitBlock = extractBetween(normalized, prefixLength + 1, prefixLength + digitLength);
    if strlength(digitBlock) > 0 && startsWith(string(digitBlock), "0")
        hasLeadingZero = true;
    end
end

function [score, label, leftMargin, rightMargin] = localFramingScore(plateImage, recognizedText, layoutHint)
    score = 0.30;
    label = "unknown";
    leftMargin = 0;
    rightMargin = 0;

    if isempty(plateImage) || strlength(string(recognizedText)) == 0
        return;
    end

    grayImage = plateImage;
    if ndims(grayImage) == 3
        grayImage = im2gray(grayImage);
    end

    grayImage = im2single(grayImage);
    if nargin >= 3 && string(layoutHint) == "two_row"
        [score, label, leftMargin, rightMargin] = localTwoRowFramingScore(grayImage);
        return;
    end

    rowStart = max(1, round(0.18 * size(grayImage, 1)));
    rowEnd = min(size(grayImage, 1), round(0.88 * size(grayImage, 1)));
    bandImage = grayImage(rowStart:rowEnd, :);

    brightMask = imbinarize(bandImage, "adaptive", ...
        "ForegroundPolarity", "bright", ...
        "Sensitivity", 0.42);
    brightMask = bwareaopen(brightMask, max(4, round(numel(brightMask) * 0.0025)));

    columnProfile = mean(brightMask, 1);
    positiveColumns = columnProfile(columnProfile > 0);
    if isempty(positiveColumns)
        label = "no_text_band";
        return;
    end
    activeColumns = columnProfile >= max(0.03, 0.45 * mean(positiveColumns));
    if ~any(activeColumns)
        label = "no_text_band";
        return;
    end

    firstColumn = find(activeColumns, 1, "first");
    lastColumn = find(activeColumns, 1, "last");
    plateWidth = size(brightMask, 2);
    leftMargin = (firstColumn - 1) / max(plateWidth, 1);
    rightMargin = (plateWidth - lastColumn) / max(plateWidth, 1);
    spanRatio = (lastColumn - firstColumn + 1) / max(plateWidth, 1);

    edgeScore = min(1, leftMargin / 0.025) * min(1, rightMargin / 0.025);
    symmetryScore = max(0, 1 - abs(leftMargin - rightMargin) / 0.12);
    spanScore = max(0, 1 - abs(spanRatio - 0.62) / 0.28);
    score = 0.45 * edgeScore + 0.30 * symmetryScore + 0.25 * spanScore;
    score = max(0, min(1, score));
    label = "text_band";
end

function [score, label, leftMargin, rightMargin] = localTwoRowFramingScore(grayImage)
    score = 0.25;
    label = "two_row_unknown";
    leftMargin = 0;
    rightMargin = 0;

    brightMask = imbinarize(grayImage, "adaptive", ...
        "ForegroundPolarity", "bright", ...
        "Sensitivity", 0.42);
    brightMask = bwareaopen(brightMask, max(4, round(numel(brightMask) * 0.0020)));

    rowProfile = mean(brightMask, 2);
    positiveRows = rowProfile(rowProfile > 0);
    if isempty(positiveRows)
        label = "two_row_no_band";
        return;
    end

    activeRows = rowProfile >= max(0.02, 0.55 * mean(positiveRows));
    rowRuns = localLogicalRuns(activeRows);
    if size(rowRuns, 1) < 2
        [score, label, leftMargin, rightMargin] = localFramingScore(grayImage, "X");
        score = 0.65 * score;
        label = "fallback_single_band";
        return;
    end

    runHeights = rowRuns(:, 2) - rowRuns(:, 1) + 1;
    [~, order] = sort(runHeights, "descend");
    rowRuns = sortrows(rowRuns(order(1:2), :), 1);

    [topMargins, topSpan] = localBandMargins(brightMask(rowRuns(1, 1):rowRuns(1, 2), :));
    [bottomMargins, bottomSpan] = localBandMargins(brightMask(rowRuns(2, 1):rowRuns(2, 2), :));
    leftMargin = mean([topMargins(1) bottomMargins(1)]);
    rightMargin = mean([topMargins(2) bottomMargins(2)]);

    marginScore = mean([ ...
        min(1, topMargins(1) / 0.02) ...
        min(1, topMargins(2) / 0.02) ...
        min(1, bottomMargins(1) / 0.02) ...
        min(1, bottomMargins(2) / 0.02)]);
    spanScore = mean([ ...
        max(0, 1 - abs(topSpan - 0.56) / 0.34) ...
        max(0, 1 - abs(bottomSpan - 0.56) / 0.34)]);
    separation = rowRuns(2, 1) - rowRuns(1, 2);
    separationScore = min(1, separation / max(0.08 * size(grayImage, 1), 1));
    balanceScore = max(0, 1 - abs(runHeights(order(1)) - runHeights(order(2))) / ...
        max(sum(runHeights(order(1:2))), 1));

    score = max(0, min(1, mean([marginScore spanScore separationScore balanceScore])));
    label = "two_row_band";
end

function reason = localSelectionReason(scoreBreakdown, recognizedText, stateInfo)
    reason = "Detector=" + sprintf("%.2f", scoreBreakdown.detector) + ...
        ", Plate=" + sprintf("%.2f", scoreBreakdown.plateEvidence) + ...
        ", OCR=" + sprintf("%.2f", scoreBreakdown.recognition) + ...
        ", Regex=" + string(scoreBreakdown.regexLabel) + ...
        ", Struct=" + string(scoreBreakdown.structureLabel) + ...
        ", Frame=" + string(scoreBreakdown.framingLabel) + ...
        ", Length=" + string(scoreBreakdown.lengthLabel) + ...
        ", State=" + string(stateInfo.name) + ...
        ", Text=" + string(recognizedText);
end

function score = localDetectorEvidenceScore(detectorCandidate)
    characterTexture = localGetCandidateField(detectorCandidate, "characterTextureScore");
    plateContrast = localGetCandidateField(detectorCandidate, "plateContrastScore");
    alignment = localGetCandidateField(detectorCandidate, "componentAlignmentScore");
    emptyPenalty = localGetCandidateField(detectorCandidate, "emptyRegionPenalty");

    score = mean([characterTexture plateContrast alignment]);
    score = max(0, min(1, score - 0.35 * emptyPenalty));
end

function value = localGetCandidateField(candidate, fieldName)
    value = 0;
    if isfield(candidate, fieldName)
        value = candidate.(fieldName);
    end
    value = max(0, min(1, double(value)));
end

function plateImage = localRecognitionPlate(recognitionMeta, fallbackPlate)
    plateImage = fallbackPlate;
    if isstruct(recognitionMeta) && isfield(recognitionMeta, "preparedPlateImage") && ...
            ~isempty(recognitionMeta.preparedPlateImage)
        plateImage = recognitionMeta.preparedPlateImage;
    elseif isstruct(recognitionMeta) && isfield(recognitionMeta, "ocrInputPlate") && ...
            ~isempty(recognitionMeta.ocrInputPlate)
        plateImage = recognitionMeta.ocrInputPlate;
    end
end

function plateImage = localDetectorPlate(inputImage, bbox)
    grayImage = inputImage;
    if ndims(grayImage) == 3
        grayImage = im2gray(grayImage);
    end

    plateImage = imcrop(grayImage, bbox);
    if isempty(plateImage)
        plateImage = grayImage;
    end
end

function bbox = localPadBBox(bbox, imageSize, paddingRatio, config)
    paddingX = bbox(3) * paddingRatio;
    paddingY = bbox(4) * paddingRatio;

    x1 = max(1, floor(bbox(1) - paddingX));
    y1 = max(1, floor(bbox(2) - paddingY));
    x2 = min(imageSize(2), ceil(bbox(1) + bbox(3) + paddingX));
    y2 = min(imageSize(1), ceil(bbox(2) + bbox(4) + paddingY));
    bbox = [x1 y1 max(1, x2 - x1) max(1, y2 - y1)];

    minCropHeight = 0;
    if nargin >= 4 && isfield(config, "rectification") && ...
            isfield(config.rectification, "minCropHeightPixels")
        minCropHeight = double(config.rectification.minCropHeightPixels);
    end
    if bbox(4) < minCropHeight
        extraHeight = minCropHeight - bbox(4);
        growTop = floor(extraHeight / 2);
        growBottom = ceil(extraHeight / 2);
        y1 = max(1, bbox(2) - growTop);
        y2 = min(imageSize(1), bbox(2) + bbox(4) - 1 + growBottom);
        bbox = [bbox(1) y1 bbox(3) max(1, y2 - y1 + 1)];
    end
end

function records = localEmptyEvaluatedCandidates()
    records = struct( ...
        "candidateIndex", {}, ...
        "bbox", {}, ...
        "rawBBox", {}, ...
        "branchName", {}, ...
        "profileName", {}, ...
        "layoutHint", {}, ...
        "scale", {}, ...
        "detectorScore", {}, ...
        "characterTextureScore", {}, ...
        "plateContrastScore", {}, ...
        "componentAlignmentScore", {}, ...
        "plateEvidenceScore", {}, ...
        "recognizedText", {}, ...
        "stateLookupText", {}, ...
        "stateInfo", {}, ...
        "textLength", {}, ...
        "detectorPlate", {}, ...
        "rectifiedPlate", {}, ...
        "ocrInputPlate", {}, ...
        "rectifyMeta", {}, ...
        "recognitionMeta", {}, ...
        "recognitionPath", {}, ...
        "recognitionScore", {}, ...
        "regexScore", {}, ...
        "stateScore", {}, ...
        "compositionScore", {}, ...
        "lengthScore", {}, ...
        "structureScore", {}, ...
        "framingScore", {}, ...
        "emptyPenalty", {}, ...
        "scoreBreakdown", {}, ...
        "finalScore", {}, ...
        "selectionReason", {});
end

function localShowEvaluatedCandidateDebug(evaluatedCandidates, config)
    if isempty(evaluatedCandidates) || ~isfield(config, "debug")
        return;
    end

    if isfield(config.debug, "printEvaluatedCandidateSummary") && ...
            config.debug.printEvaluatedCandidateSummary
        showEvaluatedCandidateSummary(evaluatedCandidates, config);
    end

    if isfield(config.debug, "showEvaluatedCandidatesFigure") && ...
            config.debug.showEvaluatedCandidatesFigure
        showEvaluatedCandidateDebugFigure(evaluatedCandidates, config);
    end

    if isfield(config.debug, "showRecognitionCandidatesFigure") && ...
            config.debug.showRecognitionCandidatesFigure
        showRecognitionCandidateDebugFigure(evaluatedCandidates, config);
    end
end

function hints = localRecognitionHints(detectorCandidate, rectifyMeta)
    layoutHint = "unknown";
    if isstruct(rectifyMeta) && isfield(rectifyMeta, "layoutHint") && strlength(string(rectifyMeta.layoutHint)) > 0
        layoutHint = string(rectifyMeta.layoutHint);
    elseif isfield(detectorCandidate, "layoutHint") && strlength(string(detectorCandidate.layoutHint)) > 0
        layoutHint = string(detectorCandidate.layoutHint);
    elseif isfield(detectorCandidate, "profileName") && string(detectorCandidate.profileName) == "two_row"
        layoutHint = "two_row";
    end

    hints = struct( ...
        "profileName", string(detectorCandidate.profileName), ...
        "layoutHint", layoutHint, ...
        "textOnlyPlate", localStructImageField(rectifyMeta, "textOnlyPlate"), ...
        "rowImages", {localStructCellField(rectifyMeta, "rowImages")}, ...
        "rowCompositePlate", localStructImageField(rectifyMeta, "rowCompositePlate"));
end

function layoutHint = localEvaluatedLayoutHint(detectorCandidate, rectifyMeta, recognitionMeta)
    layoutHint = "unknown";
    if nargin >= 3 && isstruct(recognitionMeta) && isfield(recognitionMeta, "layoutHint") && ...
            strlength(string(recognitionMeta.layoutHint)) > 0
        layoutHint = string(recognitionMeta.layoutHint);
    elseif nargin >= 2 && isstruct(rectifyMeta) && isfield(rectifyMeta, "layoutHint") && ...
            strlength(string(rectifyMeta.layoutHint)) > 0
        layoutHint = string(rectifyMeta.layoutHint);
    elseif isfield(detectorCandidate, "layoutHint") && strlength(string(detectorCandidate.layoutHint)) > 0
        layoutHint = string(detectorCandidate.layoutHint);
    elseif isfield(detectorCandidate, "profileName") && string(detectorCandidate.profileName) == "two_row"
        layoutHint = "two_row";
    end
end

function imageValue = localStructImageField(inputStruct, fieldName)
    imageValue = [];
    if isstruct(inputStruct) && isfield(inputStruct, fieldName)
        imageValue = inputStruct.(fieldName);
    end
end

function cellValue = localStructCellField(inputStruct, fieldName)
    cellValue = cell(0, 1);
    if isstruct(inputStruct) && isfield(inputStruct, fieldName) && ~isempty(inputStruct.(fieldName))
        cellValue = inputStruct.(fieldName);
    end
end

function runs = localLogicalRuns(maskVector)
    maskVector = logical(maskVector(:));
    runs = zeros(0, 2);
    if ~any(maskVector)
        return;
    end

    changes = diff([false; maskVector; false]);
    starts = find(changes == 1);
    ends = find(changes == -1) - 1;
    runs = [starts ends];
end

function [margins, spanRatio] = localBandMargins(mask)
    if isempty(mask) || ~any(mask(:))
        margins = [0 0];
        spanRatio = 0;
        return;
    end

    columnProfile = mean(mask, 1);
    activeColumns = columnProfile > 0;
    firstColumn = find(activeColumns, 1, "first");
    lastColumn = find(activeColumns, 1, "last");
    width = size(mask, 2);
    margins = [(firstColumn - 1) / max(width, 1) (width - lastColumn) / max(width, 1)];
    spanRatio = (lastColumn - firstColumn + 1) / max(width, 1);
end
