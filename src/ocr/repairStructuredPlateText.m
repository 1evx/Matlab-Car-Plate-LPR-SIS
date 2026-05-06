function [repairedText, metadata] = repairStructuredPlateText(textValue, config, options)
%REPAIRSTRUCTUREDPLATETEXT Repair OCR text using Malaysian plate structure.

    if nargin < 2 || isempty(config)
        config = defaultConfig();
    end
    if nargin < 3 || isempty(options)
        options = struct();
    end

    config = validateConfig(config);
    repairedText = upper(regexprep(string(textValue), "[^A-Z0-9]", ""));
    metadata = struct( ...
        "changed", false, ...
        "label", "", ...
        "originalText", repairedText, ...
        "selectedText", repairedText, ...
        "originalScore", 0, ...
        "selectedScore", 0, ...
        "editCount", 0);

    if strlength(repairedText) == 0
        return;
    end

    confidenceValue = localOptionNumber(options, "confidence", 1);
    layoutHint = localOptionString(options, "layoutHint", "");
    originalPriority = localStatePriority(identifyState(repairedText, config.malaysiaRules));
    originalScore = localRepairScore(repairedText, originalPriority, config, repairedText);
    metadata.originalScore = originalScore;
    metadata.selectedScore = originalScore;

    candidates = localRepairCandidates(repairedText, config, options);
    if isempty(candidates)
        return;
    end

    minGain = localOptionNumber(options, "minGain", localConfigNumber(config, "repairMinScoreGain", 0.03));
    lowConfidenceThreshold = localConfigNumber(config, "repairLowConfidenceThreshold", 0.84);
    if confidenceValue <= lowConfidenceThreshold
        minGain = max(0.005, minGain - 0.015);
    end
    if layoutHint == "two_row"
        minGain = max(0.005, minGain - 0.010);
    end

    bestText = repairedText;
    bestScore = originalScore;
    bestEdits = 0;

    for i = 1:numel(candidates)
        candidateText = candidates(i);
        if candidateText == repairedText
            continue;
        end

        candidatePriority = localStatePriority(identifyState(candidateText, config.malaysiaRules));
        editCount = localEditCount(repairedText, candidateText);
        candidateScore = localRepairScore(candidateText, candidatePriority, config, repairedText) - 0.015 * editCount + ...
            localSpecialPrefixBonus(candidateText, config);

        if candidateScore > bestScore + minGain
            bestText = candidateText;
            bestScore = candidateScore;
            bestEdits = editCount;
        end
    end

    repairedText = bestText;
    metadata.changed = repairedText ~= metadata.originalText;
    metadata.selectedText = repairedText;
    metadata.selectedScore = bestScore;
    metadata.editCount = bestEdits;
    if metadata.changed
        metadata.label = "pattern_repair";
    end
end

function candidates = localRepairCandidates(textValue, config, options)
    sourceTexts = unique([ ...
        upper(regexprep(string(textValue), "[^A-Z0-9]", "")); ...
        localOptionStringArray(options, "attemptedTexts")]);
    sourceTexts = sourceTexts(strlength(sourceTexts) > 0);

    allowSuffixLetter = isfield(config, "classification") && ...
        isfield(config.classification, "allowSuffixLetter") && ...
        logical(config.classification.allowSuffixLetter);

    candidates = strings(0, 1);

    prefixLengths = unique(double(config.classification.allowedPrefixLengths(:))).';
    digitLengths = unique(double(config.classification.allowedDigitLengths(:))).';
    suffixLengths = 0;
    if allowSuffixLetter
        suffixLengths = [0 1];
    end

    for sourceIndex = 1:numel(sourceTexts)
        sourceText = sourceTexts(sourceIndex);
        textLength = strlength(sourceText);
        candidates(end + 1, 1) = sourceText; %#ok<AGROW>

        for prefixLength = prefixLengths
            for digitLength = digitLengths
                for suffixLength = suffixLengths
                    if prefixLength + digitLength + suffixLength ~= textLength
                        continue;
                    end

                    prefixText = extractBetween(sourceText, 1, prefixLength);
                    digitText = extractBetween(sourceText, prefixLength + 1, prefixLength + digitLength);
                    suffixText = "";
                    if suffixLength > 0
                        suffixText = extractAfter(sourceText, prefixLength + digitLength);
                    end

                    candidateText = localApplyRegionMap(prefixText, "alpha") + ...
                        localApplyRegionMap(digitText, "digit") + ...
                        localApplyRegionMap(suffixText, "alpha");
                    candidates(end + 1, 1) = candidateText; %#ok<AGROW>
                end
            end
        end

        ruleCandidates = localRuleGuidedCandidates(sourceText, config);
        if ~isempty(ruleCandidates)
            candidates = [candidates; ruleCandidates(:)]; %#ok<AGROW>
        end
    end

    candidates = unique(candidates);
end

function candidates = localRuleGuidedCandidates(textValue, config)
    candidates = strings(0, 1);
    normalized = upper(regexprep(string(textValue), "[^A-Z0-9]", ""));
    if strlength(normalized) == 0
        return;
    end

    digitTail = regexp(char(normalized), "[0-9]{1,4}[A-Z]?$", "match", "once");
    if isempty(digitTail)
        return;
    end

    alphaPart = extractBefore(normalized, strlength(normalized) - strlength(string(digitTail)) + 1);
    suffixDigitTail = string(digitTail);
    prefixRules = config.malaysiaRules(strcmp(string({config.malaysiaRules.matcherType}), "prefix"));

    for i = 1:numel(prefixRules)
        token = upper(string(prefixRules(i).token));
        if strlength(token) <= 1
            continue;
        end

        similarity = localBestPrefixSimilarity(alphaPart, token);
        if similarity < 0.54
            continue;
        end

        candidate = token + suffixDigitTail;
        candidates(end + 1, 1) = candidate; %#ok<AGROW>
    end
end

function mappedText = localApplyRegionMap(textValue, regionType)
    chars = char(string(textValue));
    for i = 1:numel(chars)
        if regionType == "alpha"
            chars(i) = localAlphaChar(chars(i));
        else
            chars(i) = localDigitChar(chars(i));
        end
    end
    mappedText = string(chars);
end

function outputChar = localAlphaChar(inputChar)
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

function outputChar = localDigitChar(inputChar)
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

function similarity = localBestPrefixSimilarity(sourcePrefix, targetPrefix)
    sourcePrefix = upper(regexprep(string(sourcePrefix), "[^A-Z0-9]", ""));
    targetPrefix = upper(regexprep(string(targetPrefix), "[^A-Z0-9]", ""));
    if strlength(sourcePrefix) == 0 || strlength(targetPrefix) == 0
        similarity = 0;
        return;
    end

    sourceChar = char(sourcePrefix);
    targetChar = char(targetPrefix);
    windowLength = numel(targetChar);
    bestScore = 0;

    if numel(sourceChar) < windowLength
        score = localVisualTokenSimilarity(string(sourceChar), targetPrefix);
        similarity = score;
        return;
    end

    for startIndex = 1:(numel(sourceChar) - windowLength + 1)
        window = string(sourceChar(startIndex:(startIndex + windowLength - 1)));
        bestScore = max(bestScore, localVisualTokenSimilarity(window, targetPrefix));
    end
    similarity = bestScore;
end

function similarity = localVisualTokenSimilarity(sourceToken, targetToken)
    sourceChars = char(string(sourceToken));
    targetChars = char(string(targetToken));
    comparedLength = max(numel(sourceChars), numel(targetChars));
    if comparedLength == 0
        similarity = 0;
        return;
    end

    score = 0;
    for i = 1:max(numel(sourceChars), numel(targetChars))
        if i > numel(sourceChars) || i > numel(targetChars)
            score = score - 0.35;
            continue;
        end

        if sourceChars(i) == targetChars(i)
            score = score + 1.0;
        elseif localIsVisualConfusion(sourceChars(i), targetChars(i))
            score = score + 0.60;
        else
            score = score - 0.20;
        end
    end

    similarity = max(0, min(1, score / comparedLength));
end

function isConfusion = localIsVisualConfusion(charA, charB)
    charA = upper(char(charA));
    charB = upper(char(charB));
    confusionGroups = { ...
        'ILT1', ...
        'MNW', ...
        'UV', ...
        'BQ8', ...
        'S5', ...
        'OQD0', ...
        'Z2', ...
        'G6'};

    isConfusion = false;
    for i = 1:numel(confusionGroups)
        group = confusionGroups{i};
        if contains(group, charA) && contains(group, charB)
            isConfusion = true;
            return;
        end
    end
end

function score = localRepairScore(textValue, statePriority, config, referenceText)
    normalized = upper(regexprep(string(textValue), "[^A-Z0-9]", ""));
    if strlength(normalized) == 0
        score = -inf;
        return;
    end
    if nargin < 4
        referenceText = normalized;
    end
    referenceText = upper(regexprep(string(referenceText), "[^A-Z0-9]", ""));

    [prefixLength, digitLength, suffixLength, isOrdered] = localTextShape(normalized);
    totalLength = strlength(normalized);
    [referencePrefixLength, referenceDigitLength, ~, referenceOrdered] = localTextShape(referenceText);
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
    structureScore = 0.22 * double(isOrdered) + ...
        0.16 * double(prefixLength >= 1 && prefixLength <= 4) + ...
        0.16 * double(digitLength >= 1 && digitLength <= 4) + ...
        0.08 * double(suffixLength <= 1) + ...
        0.20 * double(prefixLength >= 2 && prefixLength <= 3) + ...
        0.18 * double(digitLength >= 3 && digitLength <= 4) + ...
        0.08 * double(suffixLength == 0);
    if localHasLeadingZeroDigitBlock(normalized)
        structureScore = max(0, structureScore - 0.35);
    end
    if referenceOrdered && referencePrefixLength >= 3 && referenceDigitLength > 0 && referenceDigitLength < 3 && ...
            prefixLength < referencePrefixLength
        structureScore = max(0, structureScore - 0.18);
    end
    stateScore = min(1, statePriority / 100);
    score = 0.42 * regexScore + 0.24 * lengthScore + 0.20 * structureScore + 0.14 * stateScore;
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
    isOrdered = (prefixLength + digitLength + suffixLength) == strlength(normalized) && ...
        suffixLength == strlength(suffix);
end

function hasLeadingZero = localHasLeadingZeroDigitBlock(normalized)
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

function priority = localStatePriority(stateInfo)
    priority = 0;
    if isstruct(stateInfo) && isfield(stateInfo, "matchedPriority")
        priority = double(stateInfo.matchedPriority);
    end
end

function editCount = localEditCount(originalText, candidateText)
    originalChars = char(string(originalText));
    candidateChars = char(string(candidateText));
    comparedLength = min(numel(originalChars), numel(candidateChars));
    editCount = sum(originalChars(1:comparedLength) ~= candidateChars(1:comparedLength)) + ...
        abs(numel(originalChars) - numel(candidateChars));
end

function value = localOptionNumber(options, fieldName, fallbackValue)
    value = fallbackValue;
    if isstruct(options) && isfield(options, fieldName) && ~isempty(options.(fieldName))
        value = double(options.(fieldName));
    end
end

function value = localOptionString(options, fieldName, fallbackValue)
    value = string(fallbackValue);
    if isstruct(options) && isfield(options, fieldName) && strlength(string(options.(fieldName))) > 0
        value = string(options.(fieldName));
    end
end

function values = localOptionStringArray(options, fieldName)
    values = strings(0, 1);
    if isstruct(options) && isfield(options, fieldName) && ~isempty(options.(fieldName))
        values = upper(regexprep(string(options.(fieldName)), "[^A-Z0-9]", ""));
        values = values(strlength(values) > 0);
    end
end

function value = localConfigNumber(config, fieldName, fallbackValue)
    value = fallbackValue;
    if isfield(config, "classification") && isfield(config.classification, fieldName)
        value = double(config.classification.(fieldName));
    end
end

function bonus = localSpecialPrefixBonus(textValue, config)
    bonus = 0;
    normalized = upper(regexprep(string(textValue), "[^A-Z0-9]", ""));
    stateInfo = identifyState(normalized, config.malaysiaRules);
    if ~isstruct(stateInfo) || ~isfield(stateInfo, "matched") || ~stateInfo.matched
        return;
    end

    matchedRule = "";
    if isfield(stateInfo, "matchedRule")
        matchedRule = string(stateInfo.matchedRule);
    end
    if strlength(matchedRule) <= 1
        return;
    end

    bonus = 0.18;
    if isfield(stateInfo, "matchedPriority") && double(stateInfo.matchedPriority) >= 90
        bonus = bonus + 0.04;
    end
end
