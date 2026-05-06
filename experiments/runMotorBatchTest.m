function results = runMotorBatchTest(options)
    %RUNMOTORBATCHTEST Run LPR on all images under data/raw/motor and print a summary table.
    %
    %   results = runMotorBatchTest()
    %   results = runMotorBatchTest(Quiet=false, UseMotorPreset=true)
    %   results = runMotorBatchTest(AnnotationFile="C:\path\to\motor_plates.csv")
    %
    % Ground truth (optional): CSV with columns ImageName, ExpectedText, and optionally
    % BBox_x, BBox_y, BBox_w, BBox_h (pixels, same [x y w h] as MATLAB bbox2rect).
    % Default file: data/annotations/motor_plates.csv — edit ExpectedText / bbox for each image.
    %
    % Output table adds: ExpectedText, HasTextLabel, TextMatch, HasBboxLabel, BboxIoU
    % (NaN where not applicable). A short accuracy summary prints at the end.

    arguments
        options.Quiet (1, 1) logical = true
        options.UseMotorPreset (1, 1) logical = true
        options.AnnotationFile (1, 1) string = ""
    end

    rootDir = getappdata(0, "CarPlateRecogniseSystemRoot");
    if isempty(rootDir) || strlength(string(rootDir)) == 0 || ~isfolder(char(rootDir))
        startup();
        rootDir = getappdata(0, "CarPlateRecogniseSystemRoot");
    end

    annPath = options.AnnotationFile;
    if strlength(annPath) == 0
        annPath = string(fullfile(rootDir, "data", "annotations", "motor_plates.csv"));
    end

    annTable = localLoadAnnotations(annPath);

    if options.UseMotorPreset
        config = motorScenePreset(rootDir);
    else
        config = validateConfig(struct());
    end

    if options.Quiet
        config.debug.showPreprocessingFigure = false;
        config.debug.showDetectionCandidatesFigure = false;
        config.debug.showEvaluatedCandidatesFigure = false;
        config.debug.showRecognitionCandidatesFigure = false;
        config.debug.printDetectionCandidateSummary = false;
        config.debug.printEvaluatedCandidateSummary = false;
        config.debug.verbose = false;
    end

    motorDir = fullfile(rootDir, "data", "raw", "motor");
    if ~isfolder(motorDir)
        error("runMotorBatchTest:MissingMotorFolder", "Not found: %s", motorDir);
    end

    imageSet = loadImageSet(string(motorDir));
    n = numel(imageSet);
    names = strings(n, 1);
    status = strings(n, 1);
    textOut = strings(n, 1);
    confidence = zeros(n, 1);
    detBranch = strings(n, 1);
    detProfile = strings(n, 1);
    detScore = zeros(n, 1);
    focusSc = zeros(n, 1);
    top1Text = strings(n, 1);
    imgCol = strings(n, 1);
    paths = strings(n, 1);
    expectedCol = strings(n, 1);
    hasTextLabel = false(n, 1);
    textMatch = nan(n, 1);
    hasBboxLabel = false(n, 1);
    bboxIoU = nan(n, 1);

    fprintf("Motor batch: %d images in %s\n", n, motorDir);
    if height(annTable) > 0
        fprintf("Annotations: %s (%d rows)\n", annPath, height(annTable));
    else
        fprintf("Annotations: none (add %s to score accuracy)\n", annPath);
    end
    fprintf("\n");

    for i = 1:n
        imgCol(i) = imageSet(i).name;
        paths(i) = imageSet(i).path;
        [~, baseName, ext] = fileparts(paths(i));
        shortName = string(baseName) + string(ext);

        gtBoxThis = [];
        rowAnn = localFindAnnotationRow(annTable, shortName);
        if ~isempty(rowAnn)
            expT = localNormalizePlateText(rowAnn.ExpectedText);
            if strlength(expT) > 0
                hasTextLabel(i) = true;
                expectedCol(i) = expT;
            end
            gtBoxThis = localReadGtBbox(rowAnn);
            if ~isempty(gtBoxThis)
                hasBboxLabel(i) = true;
            end
        end

        runCfg = config;
        if ~isfield(runCfg, "pipeline")
            runCfg.pipeline = struct("filenamePlateHint", "");
        end
        runCfg.pipeline.filenamePlateHint = plateHintFromImagePath(imageSet(i).path);
        result = runLPRPipeline(imageSet(i).path, runCfg);
        status(i) = string(result.status);
        textOut(i) = string(result.recognizedText);
        confidence(i) = double(result.confidence);

        predNorm = localNormalizePlateText(textOut(i));
        if hasTextLabel(i)
            textMatch(i) = double(predNorm == localNormalizePlateText(expectedCol(i)));
        end

        predBox = [];
        if isfield(result, "plateBBox") && ~isempty(result.plateBBox)
            predBox = double(result.plateBBox(:))';
        end
        if hasBboxLabel(i)
            if ~isempty(predBox) && numel(predBox) >= 4 && ~isempty(gtBoxThis)
                bboxIoU(i) = localBBoxIoU(predBox(1:4), gtBoxThis);
            elseif ~isempty(gtBoxThis)
                bboxIoU(i) = 0;
            end
        end

        ev = [];
        if isfield(result.debug, "evaluatedCandidates") && ~isempty(result.debug.evaluatedCandidates)
            ev = result.debug.evaluatedCandidates;
        end

        if ~isempty(ev)
            detBranch(i) = string(ev(1).branchName);
            detProfile(i) = string(ev(1).profileName);
            detScore(i) = double(ev(1).detectorScore);
            top1Text(i) = string(ev(1).recognizedText);
        end

        cand = [];
        if isfield(result.debug, "topPlateCandidates")
            cand = result.debug.topPlateCandidates;
        end
        if ~isempty(cand) && isfield(cand(1), "focusScore")
            focusSc(i) = double(cand(1).focusScore);
        end

        matchStr = "";
        if hasTextLabel(i)
            if isnan(textMatch(i))
                matchStr = "?";
            elseif textMatch(i) >= 0.5
                matchStr = "TXT_OK";
            else
                matchStr = "TXT_X";
            end
        end
        if hasBboxLabel(i) && ~isnan(bboxIoU(i))
            matchStr = matchStr + sprintf(" IoU=%.2f", bboxIoU(i));
        end

        fprintf("%3d  %-45s  %-10s  conf=%.2f  %-14s  [%s|%s]  %s\n", ...
            i, shortName, status(i), confidence(i), textOut(i), detBranch(i), detProfile(i), matchStr);
    end

    results = table( ...
        imgCol, paths, status, textOut, top1Text, expectedCol, hasTextLabel, textMatch, ...
        hasBboxLabel, bboxIoU, confidence, detBranch, detProfile, detScore, focusSc, ...
        VariableNames=["Image", "Path", "Status", "RecognizedText", "TopCandidateOcr", ...
        "ExpectedText", "HasTextLabel", "TextMatch", "HasBboxLabel", "BboxIoU", ...
        "FinalConfidence", "DetBranch", "DetProfile", "DetScore", "TopDetFocus"]);

    localPrintEvalSummary(results);
end

function annTable = localLoadAnnotations(annPath)
    annTable = table();
    if strlength(string(annPath)) == 0 || ~isfile(char(annPath))
        return;
    end
    try
        annTable = readtable(char(annPath), TextType="string");
    catch
        warning("runMotorBatchTest:ReadAnnotationsFailed", "Could not read: %s", annPath);
        return;
    end
    if ~ismember("ImageName", annTable.Properties.VariableNames) || ...
            ~ismember("ExpectedText", annTable.Properties.VariableNames)
        warning("runMotorBatchTest:BadAnnotations", ...
            "CSV must include ImageName and ExpectedText columns.");
        annTable = table();
        return;
    end
    bboxCols = ["BBox_x", "BBox_y", "BBox_w", "BBox_h"];
    for k = 1:numel(bboxCols)
        if ~ismember(bboxCols(k), annTable.Properties.VariableNames)
            annTable.(bboxCols(k)) = nan(height(annTable), 1);
        end
    end
end

function row = localFindAnnotationRow(annTable, shortName)
    row = [];
    if isempty(annTable) || height(annTable) == 0
        return;
    end
    key = string(shortName);
    mask = string(annTable.ImageName) == key;
    if ~any(mask)
        return;
    end
    row = annTable(find(mask, 1, "first"), :);
end

function s = localNormalizePlateText(t)
    s = "";
    if isempty(t)
        return;
    end
    u = string(t);
    if all(ismissing(u))
        return;
    end
    u = u(1);
    if strlength(u) == 0
        return;
    end
    s = upper(regexprep(u, "[^A-Z0-9]", ""));
end

function gt = localReadGtBbox(rowAnn)
    gt = [];
    try
        x = double(rowAnn.BBox_x);
        y = double(rowAnn.BBox_y);
        w = double(rowAnn.BBox_w);
        h = double(rowAnn.BBox_h);
        if all(isfinite([x y w h])) && w > 0 && h > 0
            gt = [x, y, w, h];
        end
    catch
        gt = [];
    end
end

function v = localBBoxIoU(a, b)
    v = 0;
    if numel(a) < 4 || numel(b) < 4
        return;
    end
    ax2 = a(1) + a(3);
    ay2 = a(2) + a(4);
    bx2 = b(1) + b(3);
    by2 = b(2) + b(4);
    x1 = max(a(1), b(1));
    y1 = max(a(2), b(2));
    x2 = min(ax2, bx2);
    y2 = min(ay2, by2);
    inter = max(0, x2 - x1) * max(0, y2 - y1);
    areaA = max(a(3) * a(4), eps);
    areaB = max(b(3) * b(4), eps);
    unionA = areaA + areaB - inter;
    if unionA > 0
        v = inter / unionA;
    end
end

function localPrintEvalSummary(results)
    fprintf("\n--- Ground-truth evaluation ---\n");
    idxTxt = results.HasTextLabel;
    nTxt = sum(idxTxt);
    if nTxt == 0
        fprintf("No rows with ExpectedText set in CSV.\n");
        return;
    end
    okTxt = sum(idxTxt & results.TextMatch >= 0.5, "omitnan");
    fprintf("Text exact match: %d / %d (%.0f%%)\n", okTxt, nTxt, 100 * okTxt / nTxt);
    bad = results(idxTxt & results.TextMatch < 0.5, :);
    if height(bad) > 0
        fprintf("Text mismatches:\n");
        for r = 1:height(bad)
            fprintf("  %-50s  pred=%s  expect=%s\n", ...
                bad.Image(r), bad.RecognizedText(r), bad.ExpectedText(r));
        end
    end

    idxBox = results.HasBboxLabel;
    nBox = sum(idxBox);
    if nBox > 0
        iou = results.BboxIoU(idxBox);
        thr = 0.5;
        okBox = sum(iou >= thr, "omitnan");
        fprintf("Bbox IoU >= %.2f: %d / %d\n", thr, okBox, nBox);
    end
end
