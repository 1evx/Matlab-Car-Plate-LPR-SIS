function output = main(imageInput, config)
%MAIN Assignment-facing entry point for the MATLAB LPR + SIS project.
%   MAIN() launches the demo GUI.
%   RESULTS = MAIN(IMAGEINPUT) runs the pipeline on one image, many images,
%   or every supported image inside a folder.

startup;

if nargin < 2 || isempty(config)
    config = defaultConfig();
else
    config = validateConfig(config);
end

if nargin < 1 || isempty(imageInput)
    output = LPRStateApp(config);
    return;
end

imagePaths = localResolveImageInputs(imageInput);
results = repmat(localEmptyBatchResult(config), numel(imagePaths), 1);

for i = 1:numel(imagePaths)
    results(i).imagePath = imagePaths(i);

    try
        pipelineResult = runLPRPipeline(imagePaths(i), config);
        pipelineResult.imagePath = imagePaths(i);
        results(i) = pipelineResult;
    catch exception
        results(i).status = "error";
        results(i).messages = "Processing failed: " + string(exception.message);
    end
end

output = results;

if nargout == 0
    disp(localBuildSummaryTable(results));
end
end

function imagePaths = localResolveImageInputs(imageInput)
if ischar(imageInput) || (isstring(imageInput) && isscalar(imageInput))
    imageInput = string(imageInput);

    if isfolder(imageInput)
        imageSet = loadImageSet(imageInput);
        imagePaths = string({imageSet.path});
    else
        imagePaths = imageInput;
    end
elseif iscell(imageInput)
    imagePaths = string(imageInput);
else
    imagePaths = string(imageInput);
end

if isempty(imagePaths)
    error("main:NoImagesFound", "No supported image files were found.");
end
end

function result = localEmptyBatchResult(config)
result = struct( ...
    "imagePath", "", ...
    "plateBBox", [], ...
    "plateScore", 0, ...
    "plateCrop", [], ...
    "rectifiedPlate", [], ...
    "recognizedText", "", ...
    "stateInfo", identifyState("", config.malaysiaRules), ...
    "confidence", 0, ...
    "recognitionPath", "matlab_ocr", ...
    "status", "initialized", ...
    "messages", strings(0, 1), ...
    "debug", struct());
end

function summaryTable = localBuildSummaryTable(results)
numResults = numel(results);

imagePath = strings(numResults, 1);
status = strings(numResults, 1);
plateText = strings(numResults, 1);
stateName = strings(numResults, 1);
confidence = zeros(numResults, 1);

for i = 1:numResults
    imagePath(i) = string(results(i).imagePath);
    status(i) = string(results(i).status);
    plateText(i) = string(results(i).recognizedText);
    stateName(i) = string(results(i).stateInfo.name);
    confidence(i) = results(i).confidence;
end

summaryTable = table(imagePath, status, plateText, stateName, confidence);
end
