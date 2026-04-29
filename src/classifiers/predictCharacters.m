function [predictions, confidences] = predictCharacters(characterImages, model, config)
%PREDICTCHARACTERS Predict characters from segmented character images.

if nargin < 3
    config = defaultConfig();
end

numCharacters = numel(characterImages);
predictions = strings(1, numCharacters);
confidences = zeros(1, numCharacters);

if numCharacters == 0
    return;
end

featureMatrix = zeros(numCharacters, numel(extractCharacterFeatures(characterImages{1}, config)));
for i = 1:numCharacters
    featureMatrix(i, :) = extractCharacterFeatures(characterImages{i}, config);
end

if isfield(model, "classifier") && ~isempty(model.classifier)
    [predictions, scores] = predict(model.classifier, featureMatrix);
    predictions = string(predictions).';
    for i = 1:size(scores, 1)
        confidences(i) = max(scores(i, :));
    end
    return;
end

for i = 1:numCharacters
    distances = sum((model.features - featureMatrix(i, :)).^2, 2);
    [sortedDistances, sortIdx] = sort(distances, "ascend");
    predictions(i) = model.labels(sortIdx(1));
    confidences(i) = 1 / (1 + sortedDistances(1));
end
end

