function showRecognitionCandidateDebugFigure(evaluatedCandidates, config)
%SHOWRECOGNITIONCANDIDATEDEBUGFIGURE Show OCR character predictions for top evaluated candidates.

    displayCount = min(localDisplayCount(config), numel(evaluatedCandidates));
    if displayCount == 0
        return;
    end

    records = evaluatedCandidates(1:displayCount);
    maxCharacters = max(arrayfun(@(r) localPreparedGlyphCount(r), records));
    if maxCharacters == 0
        return;
    end
    totalCols = max(2, maxCharacters + 1);

    figureHandle = findall(groot, "Type", "figure", "Tag", "LPRRecognitionCandidatesDebug");
    if isempty(figureHandle) || ~isvalid(figureHandle(1))
        figureHandle = figure( ...
            "Name", "LPR Recognition Candidates", ...
            "NumberTitle", "off", ...
            "Tag", "LPRRecognitionCandidatesDebug");
    else
        figureHandle = figureHandle(1);
        clf(figureHandle);
        figure(figureHandle);
    end

    tiledlayout(figureHandle, displayCount, totalCols, ...
        "TileSpacing", "compact", ...
        "Padding", "compact");

    for rowIdx = 1:displayCount
        record = records(rowIdx);
        nexttile;
        imshow(localPlatePreview(record));
        title(localPlateTitle(record, rowIdx), ...
            "FontSize", 8, ...
            "Interpreter", "none");

        preparedGlyphs = localPreparedGlyphs(record);
        topCandidates = localRecognitionTopCandidates(record);
        for charIdx = 1:maxCharacters
            nexttile;
            if charIdx <= numel(preparedGlyphs) && ~isempty(preparedGlyphs{charIdx})
                imshow(preparedGlyphs{charIdx});
                title(localGlyphTitle(topCandidates, charIdx), ...
                    "FontSize", 8, ...
                    "Interpreter", "none");
            else
                imshow(false(48, 24));
                title(sprintf("Char %d\n<none>", charIdx), ...
                    "FontSize", 8, ...
                    "Interpreter", "none");
            end
        end
    end
end

function count = localDisplayCount(config)
    count = 4;
    if isfield(config, "debug") && isfield(config.debug, "maxRecognitionCandidatesToDisplay")
        count = max(1, round(double(config.debug.maxRecognitionCandidatesToDisplay)));
    end
end

function count = localPreparedGlyphCount(record)
    count = numel(localPreparedGlyphs(record));
end

function preparedGlyphs = localPreparedGlyphs(record)
    preparedGlyphs = {};
    if isfield(record, "recognitionMeta") && isstruct(record.recognitionMeta) && ...
            isfield(record.recognitionMeta, "preparedGlyphs")
        preparedGlyphs = record.recognitionMeta.preparedGlyphs;
    end
end

function topCandidates = localRecognitionTopCandidates(record)
    topCandidates = {};
    if isfield(record, "recognitionMeta") && isstruct(record.recognitionMeta) && ...
            isfield(record.recognitionMeta, "topCandidates")
        topCandidates = record.recognitionMeta.topCandidates;
    end
end

function imageOut = localPlatePreview(record)
    if isfield(record, "ocrInputPlate") && ~isempty(record.ocrInputPlate)
        imageOut = record.ocrInputPlate;
    elseif isfield(record, "rectifiedPlate") && ~isempty(record.rectifiedPlate)
        imageOut = record.rectifiedPlate;
    else
        imageOut = zeros(40, 120, "uint8");
    end
end

function titleText = localPlateTitle(record, rankIdx)
    titleText = sprintf([ ...
        '#%d raw#%d %.3f\n' ...
        '%s | chars %d | rec %.2f\n' ...
        'text %s'], ...
        rankIdx, ...
        record.candidateIndex, ...
        record.finalScore, ...
        char(string(record.recognitionPath)), ...
        numel(localRecognitionTopCandidates(record)), ...
        record.recognitionScore, ...
        char(localShortText(record.recognizedText)));
end

function titleText = localGlyphTitle(topCandidates, charIdx)
    if charIdx > numel(topCandidates) || isempty(topCandidates{charIdx})
        titleText = sprintf("Char %d\n<none>", charIdx);
        return;
    end

    candidateInfo = topCandidates{charIdx};
    topLabels = candidateInfo.labels(:);
    topScores = candidateInfo.scores(:);
    entryCount = min(3, numel(topLabels));
    lines = strings(entryCount, 1);
    for i = 1:entryCount
        lines(i) = topLabels(i) + " " + sprintf("%.2f", topScores(i));
    end
    titleText = sprintf("Char %d\n%s", charIdx, char(join(lines, " | ")));
end

function textOut = localShortText(textIn)
    textOut = string(textIn);
    if strlength(textOut) == 0
        textOut = "<empty>";
    elseif strlength(textOut) > 12
        textOut = extractBefore(textOut, 13) + "...";
    end
end
