function result = runLPRPipeline(imageInput, config)
    %RUNLPRPIPELINE End-to-end Malaysian license plate recognition pipeline.

    config = validateConfig(config);
    inputImage = localReadImage(imageInput);

    result = struct( ...
        "plateBBox", [], ...
        "plateScore", 0, ...
        "plateCrop", [], ...
        "rectifiedPlate", [], ...
        "recognizedText", "", ...
        "stateInfo", identifyState("", config.malaysiaRules), ...
        "confidence", 0, ...
        "recognitionPath", "matlab_ocr", ...
        "status", "initialized", ...
        "messages", strings(0,1), ...
        "debug", struct());

    % Stage 1: grayscale conversion, denoising, and contrast enhancement.
    [~, preprocessMeta] = preprocessImage(inputImage, config);
    localShowPreprocessingDebugFigure(preprocessMeta, config);

    % Stage 2: locate the most likely plate region in the full vehicle image.
    [plateBBox, plateMeta] = detectPlateRegion(preprocessMeta.grayImage, config);
    result.debug.gray = preprocessMeta.grayImage;
    result.debug.denoised = preprocessMeta.denoisedImage;
    result.debug.enhanced = preprocessMeta.preprocessedImage;
    result.debug.edgeMask = plateMeta.edgeMask;
    result.debug.closedMask = plateMeta.closedMask;
    result.debug.openedMask = plateMeta.openedMask;
    result.debug.dilatedMask = plateMeta.dilatedMask;
    result.debug.plateMask = plateMeta.plateMask;
    result.debug.priorityEdgeMask = plateMeta.priorityEdgeMask;
    result.debug.darkPlateMask = plateMeta.darkPlateMask;
    result.debug.textClusterMask = plateMeta.textClusterMask;
    result.debug.multiScaleEdgeMask = plateMeta.multiScaleEdgeMask;
    result.debug.plateCandidates = plateMeta.candidates;
    result.debug.topPlateCandidates = plateMeta.topCandidates;

    if isempty(plateBBox)
        result.status = "plate_not_found";
        result.messages(end+1) = "No plate candidate passed the configured filters.";
        return;
    end

    [selectedCandidateIndex, evaluatedCandidates] = rerankPlateCandidates(inputImage, plateMeta.topCandidates, config);
    if isempty(selectedCandidateIndex) || isempty(evaluatedCandidates)
        result.status = "plate_not_found";
        result.messages(end+1) = "Plate candidates were generated, but none survived reranking.";
        return;
    end

    selectedCandidate = evaluatedCandidates(selectedCandidateIndex);
    plateBBox = selectedCandidate.bbox;
    rectifiedPlate = selectedCandidate.rectifiedPlate;
    rectifyMeta = selectedCandidate.rectifyMeta;
    recognizedText = selectedCandidate.recognizedText;
    recognitionMeta = selectedCandidate.recognitionMeta;
    stateInfo = selectedCandidate.stateInfo;

    result.plateBBox = plateBBox;
    result.plateScore = selectedCandidate.detectorScore;
    result.plateCrop = rectifyMeta.croppedPlate;
    result.rectifiedPlate = rectifiedPlate;
    result.recognizedText = string(recognizedText);
    result.stateInfo = stateInfo;
    result.confidence = selectedCandidate.finalScore;
    result.recognitionPath = string(selectedCandidate.recognitionPath);
    recognitionSucceeded = localRecognitionSucceeded(recognitionMeta, recognizedText);
    if recognitionSucceeded
        result.status = "ok";
        result.messages(end+1) = "Pipeline completed successfully.";
    else
        result.status = "ocr_failed";
        result.messages(end+1) = "Plate candidate was detected, but MATLAB OCR returned empty text.";
    end
    result.messages(end+1) = "Selected candidate " + selectedCandidate.candidateIndex + ...
        " after regex/state reranking.";

    result.debug.rectifiedBinaryMask = rectifyMeta.binaryMask;
    result.debug.rectifiedAngle = rectifyMeta.angle;
    result.debug.recognition = recognitionMeta;
    result.debug.ocrInputPlate = selectedCandidate.ocrInputPlate;
    result.debug.evaluatedCandidates = evaluatedCandidates;
    result.debug.selectedCandidateIndex = selectedCandidateIndex;
    result.debug.selectedCandidateReason = selectedCandidate.selectionReason;
    result.debug.selectedRecognitionPath = string(selectedCandidate.recognitionPath);
    result.debug.overlay = drawResults(inputImage, result);
end

function isSuccess = localRecognitionSucceeded(recognitionMeta, recognizedText)
    isSuccess = strlength(string(recognizedText)) > 0;
    if isstruct(recognitionMeta) && isfield(recognitionMeta, "matlabOcr") && ...
            isstruct(recognitionMeta.matlabOcr) && isfield(recognitionMeta.matlabOcr, "success")
        isSuccess = isSuccess && logical(recognitionMeta.matlabOcr.success);
    end
end

function image = localReadImage(imageInput)
    if isstring(imageInput) || ischar(imageInput)
        image = imread(imageInput);
    else
        image = imageInput;
    end

    if ndims(image) == 2
        image = repmat(image, 1, 1, 3);
    end
end

function localShowPreprocessingDebugFigure(preprocessMeta, config)
    if ~isfield(config, "debug") || ~isfield(config.debug, "showPreprocessingFigure") || ...
            ~config.debug.showPreprocessingFigure
        return;
    end

    showPreprocessingDebugFigure(preprocessMeta);
end
