function showEvaluatedCandidateDebugFigure(evaluatedCandidates, config)
%SHOWEVALUATEDCANDIDATEDEBUGFIGURE Show top reranked candidates with OCR details.

    displayCount = min(localDisplayCount(config), numel(evaluatedCandidates));
    records = evaluatedCandidates(1:displayCount);

    figureHandle = findall(groot, "Type", "figure", "Tag", "LPREvaluatedCandidatesDebug");
    if isempty(figureHandle) || ~isvalid(figureHandle(1))
        figureHandle = figure( ...
            "Name", "LPR Evaluated Candidates", ...
            "NumberTitle", "off", ...
            "Tag", "LPREvaluatedCandidatesDebug");
    else
        figureHandle = figureHandle(1);
        clf(figureHandle);
        figure(figureHandle);
    end

    tileCols = min(4, displayCount);
    tileRows = ceil(displayCount / tileCols);
    tiledlayout(figureHandle, tileRows, tileCols, ...
        "TileSpacing", "compact", ...
        "Padding", "compact");

    for i = 1:displayCount
        nexttile;
        displayImage = localDisplayImage(records(i));
        imshow(displayImage);
        title(sprintf([ ...
            '#%d %.3f | Det %.2f Plate %.2f OCR %.2f\n' ...
            'Regex %.2f | State %.2f | Len %.2f\n' ...
            'Struct %.2f | Frame %.2f\n' ...
            'Left raw detector | Right OCR rectified\n' ...
            '%s | %s | %s\n' ...
            'Text %s'], ...
            i, ...
            records(i).finalScore, ...
            records(i).detectorScore, ...
            records(i).plateEvidenceScore, ...
            records(i).recognitionScore, ...
            records(i).regexScore, ...
            records(i).stateScore, ...
            records(i).lengthScore, ...
            records(i).structureScore, ...
            records(i).framingScore, ...
            char(string(records(i).recognitionPath)), ...
            char(string(records(i).profileName)), ...
            char(string(records(i).branchName)), ...
            char(localDisplayText(records(i).recognizedText))), ...
            "FontSize", 8, ...
            "Interpreter", "none");
    end
end

function imageOut = localDisplayImage(record)
    detectorImage = localDetectorImage(record);
    ocrImage = localOcrImage(record);

    detectorImage = localPrepareDisplayImage(detectorImage);
    ocrImage = localPrepareDisplayImage(ocrImage);

    targetHeight = max(size(detectorImage, 1), size(ocrImage, 1));
    detectorImage = localResizeToHeight(detectorImage, targetHeight);
    ocrImage = localResizeToHeight(ocrImage, targetHeight);

    spacer = uint8(235 * ones(targetHeight, 10, 3));
    imageOut = cat(2, detectorImage, spacer, ocrImage);
end

function imageOut = localDetectorImage(record)
    if isfield(record, "detectorPlate") && ~isempty(record.detectorPlate)
        imageOut = record.detectorPlate;
    elseif isfield(record, "rectifiedPlate") && ~isempty(record.rectifiedPlate)
        imageOut = record.rectifiedPlate;
    else
        imageOut = zeros(40, 120, "uint8");
    end
end

function imageOut = localOcrImage(record)
    if isfield(record, "ocrInputPlate") && ~isempty(record.ocrInputPlate)
        imageOut = record.ocrInputPlate;
    elseif isfield(record, "rectifiedPlate") && ~isempty(record.rectifiedPlate)
        imageOut = record.rectifiedPlate;
    else
        imageOut = zeros(40, 120, "uint8");
    end
end

function imageOut = localPrepareDisplayImage(imageIn)
    if isempty(imageIn)
        imageOut = zeros(40, 120, 3, "uint8");
        return;
    end

    if islogical(imageIn)
        imageOut = uint8(imageIn) * 255;
    elseif isa(imageIn, "uint8")
        imageOut = imageIn;
    elseif isfloat(imageIn)
        imageOut = im2uint8(mat2gray(imageIn));
    else
        imageOut = uint8(imageIn);
    end

    if ndims(imageOut) == 2
        imageOut = repmat(imageOut, 1, 1, 3);
    end
end

function imageOut = localResizeToHeight(imageIn, targetHeight)
    if size(imageIn, 1) == targetHeight
        imageOut = imageIn;
        return;
    end

    targetWidth = max(1, round(size(imageIn, 2) * targetHeight / max(size(imageIn, 1), 1)));
    imageOut = imresize(imageIn, [targetHeight targetWidth], "nearest");
end

function textOut = localDisplayText(textIn)
    textOut = string(textIn);
    if strlength(textOut) == 0
        textOut = "<empty>";
        return;
    end

    if strlength(textOut) > 14
        textOut = extractBefore(textOut, 15) + "...";
    end
end

function count = localDisplayCount(config)
    count = 12;
    if isfield(config, "debug") && isfield(config.debug, "maxEvaluatedCandidatesToDisplay")
        count = max(1, round(double(config.debug.maxEvaluatedCandidatesToDisplay)));
    end
end
