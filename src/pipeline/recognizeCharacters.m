function [recognizedText, metadata] = recognizeCharacters(characterImages, config, plateImage)
    %RECOGNIZECHARACTERS Recognize segmented plate characters with feature classification.

    if nargin < 3
        plateImage = [];
    end

    config = validateConfig(config);

    if isempty(characterImages)
        recognizedText = "";
        metadata = localEmptyMetadata(plateImage);
        return;
    end

    model = localLoadOrTrainModel(config);
    [scoreMatrix, labels, featureMatrix, topCandidates, rawPredictions, rawConfidences] = ...
        localScoreCharacters(characterImages, model, config);

    [parsedText, parsedConfidences, parsedPattern] = ...
        localParsePlatePattern(scoreMatrix, labels, config);
    if strlength(parsedText) == 0
        recognizedText = join(rawPredictions, "");
        confidences = rawConfidences(:);
    else
        recognizedText = parsedText;
        confidences = parsedConfidences(:);
    end

    stateLookupText = localDeriveStateLookupText(scoreMatrix, labels, config, recognizedText);
    metadata = struct( ...
        "predictions", string(regexp(char(recognizedText), ".", "match")).', ...
        "confidences", confidences, ...
        "rawPredictions", rawPredictions(:), ...
        "rawConfidences", rawConfidences(:), ...
        "stateLookupText", stateLookupText, ...
        "parsedText", parsedText, ...
        "parsedPattern", parsedPattern, ...
        "topCandidates", {topCandidates}, ...
        "featureVectors", featureMatrix, ...
        "method", "feature_classifier", ...
        "plateImage", plateImage);
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
        "featureVectors", zeros(0, 0), ...
        "method", "none", ...
        "plateImage", plateImage);
end

function model = localLoadOrTrainModel(config)
    modelPath = string(config.classification.trainedModelPath);
    requiresRetrain = ~isfile(modelPath);
    if requiresRetrain && ~config.classification.autoTrainIfMissing
        error("recognizeCharacters:MissingModel", ...
            "Feature classifier model was not found: %s", modelPath);
    end

    if requiresRetrain
        trainCharacterClassifier(modelPath, config);
    end

    loaded = load(modelPath, "model");
    model = loaded.model;
    expectedFeatureLength = numel( ...
        extractCharacterFeatures(false(config.features.characterImageSize), config));
    modelFeatureLength = size(model.features, 2);

    if modelFeatureLength ~= expectedFeatureLength && config.classification.autoTrainIfMissing
        trainCharacterClassifier(modelPath, config);
        loaded = load(modelPath, "model");
        model = loaded.model;
    end
end

function [scoreMatrix, labels, featureMatrix, topCandidates, rawPredictions, rawConfidences] = ...
        localScoreCharacters(characterImages, model, config)

    labels = localCharacterLabels(config);
    numCharacters = numel(characterImages);
    featureLength = numel(extractCharacterFeatures(false(config.features.characterImageSize), config));
    featureMatrix = zeros(numCharacters, featureLength);
    scoreMatrix = zeros(numCharacters, numel(labels));
    topCandidates = cell(numCharacters, 1);
    rawPredictions = strings(numCharacters, 1);
    rawConfidences = zeros(numCharacters, 1);

    for i = 1:numCharacters
        preparedGlyph = localPrepareGlyph(characterImages{i}, config);
        featureMatrix(i, :) = extractCharacterFeatures(preparedGlyph, config);
        scoreMatrix(i, :) = localClassScores(featureMatrix(i, :), model, labels);

        [sortedScores, sortIdx] = sort(scoreMatrix(i, :), "descend");
        topCount = min(config.classification.topCandidatesPerCharacter, numel(sortIdx));
        topCandidates{i} = struct( ...
            "labels", labels(sortIdx(1:topCount)), ...
            "scores", sortedScores(1:topCount));

        rawPredictions(i) = labels(sortIdx(1));
        rawConfidences(i) = sortedScores(1);
    end
end

function labels = localCharacterLabels(config)
    characterSet = char(string(config.classification.characterSet));
    labels = string(cellstr(characterSet.'));
end

function preparedGlyph = localPrepareGlyph(glyphImage, config)
    if isempty(glyphImage)
        preparedGlyph = false(config.features.characterImageSize);
        return;
    end

    preparedGlyph = glyphImage;
    if ~islogical(preparedGlyph)
        preparedGlyph = imbinarize(im2gray(preparedGlyph));
    end

    preparedGlyph = logical(preparedGlyph);
    if mean(preparedGlyph(:)) > 0.5
        preparedGlyph = ~preparedGlyph;
    end

    preparedGlyph = bwareaopen(preparedGlyph, 6);
    props = regionprops(preparedGlyph, "BoundingBox", "Area");
    if isempty(props)
        preparedGlyph = false(config.features.characterImageSize);
        return;
    end

    [~, idx] = max([props.Area]);
    tightBox = ceil(props(idx).BoundingBox);
    preparedGlyph = imcrop(preparedGlyph, tightBox);
    if isempty(preparedGlyph)
        preparedGlyph = false(config.features.characterImageSize);
        return;
    end

    preparedGlyph = padarray(preparedGlyph, [2 2], 0, "both");
end

function scores = localClassScores(featureVector, model, labels)
    scores = zeros(1, numel(labels));

    if isfield(model, "classifier") && ~isempty(model.classifier)
        [~, classifierScores] = predict(model.classifier, featureVector);
        classNames = string(model.classifier.ClassNames);
        for i = 1:numel(classNames)
            labelIdx = find(labels == classNames(i), 1);
            if ~isempty(labelIdx)
                scores(labelIdx) = classifierScores(i);
            end
        end
        scores = localNormalizeScores(scores);
        return;
    end

    for i = 1:numel(labels)
        labelMask = model.labels == labels(i);
        labelFeatures = model.features(labelMask, :);
        if isempty(labelFeatures)
            continue;
        end

        distances = sum((labelFeatures - featureVector).^2, 2);
        scores(i) = max(1 ./ (1 + distances));
    end

    scores = localNormalizeScores(scores);
end

function scores = localNormalizeScores(scores)
    scores = max(0, double(scores));
    totalScore = sum(scores);
    if totalScore > 0
        scores = scores / totalScore;
    end
end

function [parsedText, parsedConfidences, parsedPattern] = localParsePlatePattern(scoreMatrix, labels, config)
    parsedText = "";
    parsedConfidences = zeros(0, 1);
    parsedPattern = "";
    numCharacters = size(scoreMatrix, 1);
    bestScore = -inf;

    for prefixLength = config.classification.allowedPrefixLengths
        for digitLength = config.classification.allowedDigitLengths
            for suffixLength = localSuffixOptions(config)
                if prefixLength + digitLength + suffixLength ~= numCharacters
                    continue;
                end

                [candidateText, candidateConfidences, candidateScore] = localScorePattern( ...
                    scoreMatrix, labels, prefixLength, digitLength, suffixLength, config);

                if candidateScore > bestScore
                    bestScore = candidateScore;
                    parsedText = candidateText;
                    parsedConfidences = candidateConfidences;
                    parsedPattern = sprintf("L%d-D%d-S%d", ...
                        prefixLength, digitLength, suffixLength);
                end
            end
        end
    end

    if bestScore < config.classification.minimumConfidence
        parsedText = "";
        parsedConfidences = zeros(0, 1);
        parsedPattern = "";
    end
end

function suffixOptions = localSuffixOptions(config)
    if config.classification.allowSuffixLetter
        suffixOptions = [0 1];
    else
        suffixOptions = 0;
    end
end

function [candidateText, candidateConfidences, candidateScore] = ...
        localScorePattern(scoreMatrix, labels, prefixLength, digitLength, suffixLength, config)

    numCharacters = size(scoreMatrix, 1);
    candidateChars = strings(1, numCharacters);
    candidateConfidences = zeros(numCharacters, 1);
    candidateScore = 0;

    for position = 1:numCharacters
        if position <= prefixLength
            allowedClass = "letter";
        elseif position <= prefixLength + digitLength
            allowedClass = "digit";
        else
            allowedClass = "letter";
        end

        [bestLabel, bestScore] = localBestAllowedLabel( ...
            scoreMatrix(position, :), labels, allowedClass, config);
        candidateChars(position) = bestLabel;
        candidateConfidences(position) = bestScore;
        candidateScore = candidateScore + bestScore;
    end

    candidateText = join(candidateChars, "");
    candidateScore = candidateScore / max(numCharacters, 1);

    if strlength(candidateText) == 0
        candidateScore = -inf;
    elseif ~isempty(regexp(candidateText, "^[A-Z]{1,3}[0-9]{1,4}[A-Z]?$", "once"))
        candidateScore = candidateScore + 0.02;
    else
        candidateScore = candidateScore - config.classification.invalidPatternPenalty;
    end
end

function [bestLabel, bestScore] = localBestAllowedLabel(scoreRow, labels, allowedClass, config)
    switch allowedClass
        case "letter"
            allowedMask = arrayfun(@(x) all(isstrprop(char(x), "alpha")), labels);
        otherwise
            allowedMask = arrayfun(@(x) all(isstrprop(char(x), "digit")), labels);
    end

    allowedLabels = labels(allowedMask);
    bestLabel = "";
    bestScore = -inf;

    for i = 1:numel(allowedLabels)
        candidateLabel = allowedLabels(i);
        score = localScoreLabelWithAmbiguity(scoreRow, labels, candidateLabel, config);
        if score > bestScore
            bestScore = score;
            bestLabel = candidateLabel;
        end
    end
end

function score = localScoreLabelWithAmbiguity(scoreRow, labels, targetLabel, config)
    targetIdx = find(labels == targetLabel, 1);
    if isempty(targetIdx)
        score = -inf;
        return;
    end

    score = scoreRow(targetIdx);
    alternateLabel = localAmbiguousPartner(targetLabel);
    if strlength(alternateLabel) == 0
        return;
    end

    alternateIdx = find(labels == alternateLabel, 1);
    if isempty(alternateIdx)
        return;
    end

    score = max(score, scoreRow(alternateIdx) - config.classification.ambiguityPenalty);
end

function alternateLabel = localAmbiguousPartner(label)
    switch char(label)
        case 'O'
            alternateLabel = "0";
        case '0'
            alternateLabel = "O";
        case 'I'
            alternateLabel = "1";
        case '1'
            alternateLabel = "I";
        case 'S'
            alternateLabel = "5";
        case '5'
            alternateLabel = "S";
        case 'B'
            alternateLabel = "8";
        case '8'
            alternateLabel = "B";
        case 'Z'
            alternateLabel = "2";
        case '2'
            alternateLabel = "Z";
        case 'G'
            alternateLabel = "6";
        case '6'
            alternateLabel = "G";
        otherwise
            alternateLabel = "";
    end
end

function stateLookupText = localDeriveStateLookupText(scoreMatrix, labels, config, recognizedText)
    stateLookupText = string(recognizedText);
    if strlength(stateLookupText) > 0
        return;
    end

    prefixLength = min(3, size(scoreMatrix, 1));
    prefixChars = strings(1, prefixLength);
    for i = 1:prefixLength
        [prefixChars(i), ~] = localBestAllowedLabel( ...
            scoreMatrix(i, :), labels, "letter", config);
    end
    stateLookupText = join(prefixChars, "");
end
