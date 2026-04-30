function resultsTable = evaluateFailedImageSet(manifestInput, config)
%EVALUATEFAILEDIMAGESET Run the LPR pipeline over a labeled failure set.

if nargin < 2 || isempty(config)
    config = defaultConfig();
end
config = validateConfig(config);

manifestTable = localReadManifest(manifestInput);
requiredColumns = ["image_path" "expected_text"];
for i = 1:numel(requiredColumns)
    if ~ismember(requiredColumns(i), string(manifestTable.Properties.VariableNames))
        error("evaluateFailedImageSet:MissingColumn", ...
            "Manifest must contain column '%s'.", requiredColumns(i));
    end
end

rowCount = height(manifestTable);
records = repmat(struct( ...
    "ImagePath", "", ...
    "ExpectedText", "", ...
    "RecognizedText", "", ...
    "Status", "", ...
    "DetectedProfile", "", ...
    "DetectedLayout", "", ...
    "SelectedVariant", "", ...
    "Confidence", 0, ...
    "Matched", false), rowCount, 1);

for i = 1:rowCount
    imagePath = string(manifestTable.image_path(i));
    expectedText = upper(regexprep(string(manifestTable.expected_text(i)), "[^A-Z0-9]", ""));
    result = runLPRPipeline(imagePath, config);

    detectedProfile = "";
    detectedLayout = "";
    selectedVariant = "";
    if isfield(result, "debug") && isfield(result.debug, "evaluatedCandidates") && ...
            ~isempty(result.debug.evaluatedCandidates) && ...
            isfield(result.debug, "selectedCandidateIndex") && ~isempty(result.debug.selectedCandidateIndex)
        selectedCandidate = result.debug.evaluatedCandidates(result.debug.selectedCandidateIndex);
        detectedProfile = string(selectedCandidate.profileName);
        detectedLayout = string(selectedCandidate.layoutHint);
        if isfield(selectedCandidate, "recognitionMeta") && ...
                isfield(selectedCandidate.recognitionMeta, "ocrInputName")
            selectedVariant = string(selectedCandidate.recognitionMeta.ocrInputName);
        end
    elseif isfield(result, "debug") && isfield(result.debug, "rectifiedLayoutHint")
        detectedLayout = string(result.debug.rectifiedLayoutHint);
    end

    recognizedText = upper(regexprep(string(result.recognizedText), "[^A-Z0-9]", ""));
    records(i) = struct( ...
        "ImagePath", imagePath, ...
        "ExpectedText", expectedText, ...
        "RecognizedText", recognizedText, ...
        "Status", string(result.status), ...
        "DetectedProfile", detectedProfile, ...
        "DetectedLayout", detectedLayout, ...
        "SelectedVariant", selectedVariant, ...
        "Confidence", double(result.confidence), ...
        "Matched", recognizedText == expectedText);
end

resultsTable = struct2table(records);
end

function manifestTable = localReadManifest(manifestInput)
if istable(manifestInput)
    manifestTable = manifestInput;
elseif isstring(manifestInput) || ischar(manifestInput)
    manifestTable = readtable(manifestInput, "TextType", "string");
else
    error("evaluateFailedImageSet:InvalidInput", ...
        "Manifest input must be a table or a file path.");
end
end
