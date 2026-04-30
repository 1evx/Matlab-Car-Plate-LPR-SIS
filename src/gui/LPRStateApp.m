classdef LPRStateApp < handle
    %LPRSTATEAPP Programmatic GUI for MATLAB LPR and state identification demos.

    properties
        Config
        CurrentImage
        CurrentImagePath string = ""
        LastResult struct = struct()
    end

    properties (Access = private)
        UIFigure matlab.ui.Figure
        RootLayout matlab.ui.container.GridLayout
        LeftLayout matlab.ui.container.GridLayout
        RightLayout matlab.ui.container.GridLayout
        ButtonLayout matlab.ui.container.GridLayout
        LoadButton matlab.ui.control.Button
        RunButton matlab.ui.control.Button
        ResetButton matlab.ui.control.Button
        InputAxes matlab.ui.control.UIAxes
        PlateAxes matlab.ui.control.UIAxes
        DebugAxes matlab.ui.control.UIAxes
        ImagePathField matlab.ui.control.EditField
        PlateTextField matlab.ui.control.EditField
        StateField matlab.ui.control.EditField
        CategoryField matlab.ui.control.EditField
        ConfidenceField matlab.ui.control.EditField
        StatusArea matlab.ui.control.TextArea
    end

    methods
        function app = LPRStateApp(config, options)
            if nargin < 1 || isempty(config)
                config = defaultConfig();
            end
            if nargin < 2
                options = struct();
            end

            app.Config = validateConfig(config);
            app.buildUI(options);
            app.resetView();
        end

        function delete(app)
            if ~isempty(app.UIFigure) && isvalid(app.UIFigure)
                delete(app.UIFigure);
            end
        end
    end

    methods (Access = private)
        function buildUI(app, options)
            visible = app.Config.gui.defaultVisible;
            if isfield(options, "Visible")
                visible = options.Visible;
            end

            app.UIFigure = uifigure( ...
                "Name", app.Config.gui.windowTitle, ...
                "Position", [100 100 1280 760], ...
                "Visible", visible);

            app.RootLayout = uigridlayout(app.UIFigure, [1 2]);
            app.RootLayout.ColumnWidth = {'2.5x', '1.2x'};

            app.LeftLayout = uigridlayout(app.RootLayout, [3 1]);
            app.LeftLayout.RowHeight = {'1x', '0.65x', '0.65x'};

            app.InputAxes = uiaxes(app.LeftLayout);
            title(app.InputAxes, "Input / Overlay")
            axis(app.InputAxes, "off")

            app.PlateAxes = uiaxes(app.LeftLayout);
            title(app.PlateAxes, "Rectified Plate")
            axis(app.PlateAxes, "off")

            app.DebugAxes = uiaxes(app.LeftLayout);
            title(app.DebugAxes, "OCR Input")
            axis(app.DebugAxes, "off")

            app.RightLayout = uigridlayout(app.RootLayout, [14 1]);
            app.RightLayout.RowHeight = {26, 26, 110, 26, 26, 26, 26, 26, 26, 26, 26, 26, '2x'};

            uilabel(app.RightLayout, "Text", "Image Path");
            app.ImagePathField = uieditfield(app.RightLayout, "text", "Editable", "off");

            app.ButtonLayout = uigridlayout(app.RightLayout, [1 3]);
            app.ButtonLayout.ColumnWidth = {'1x', '1x', '1x'};

            app.LoadButton = uibutton(app.ButtonLayout, "push", ...
                "Text", "Load Image", ...
                "ButtonPushedFcn", @(~, ~) app.onLoadImage());
            app.RunButton = uibutton(app.ButtonLayout, "push", ...
                "Text", "Run LPR", ...
                "ButtonPushedFcn", @(~, ~) app.onRunPipeline());
            app.ResetButton = uibutton(app.ButtonLayout, "push", ...
                "Text", "Reset", ...
                "ButtonPushedFcn", @(~, ~) app.onReset());

            uilabel(app.RightLayout, "Text", "Recognized Plate");
            app.PlateTextField = uieditfield(app.RightLayout, "text", "Editable", "off");

            uilabel(app.RightLayout, "Text", "State");
            app.StateField = uieditfield(app.RightLayout, "text", "Editable", "off");

            uilabel(app.RightLayout, "Text", "Category");
            app.CategoryField = uieditfield(app.RightLayout, "text", "Editable", "off");

            uilabel(app.RightLayout, "Text", "Confidence");
            app.ConfidenceField = uieditfield(app.RightLayout, "text", "Editable", "off");

            uilabel(app.RightLayout, "Text", "Log & Status");
            app.StatusArea = uitextarea(app.RightLayout, "Editable", "off", "FontSize", 12);
        end

        function onLoadImage(app)
            [fileName, folderPath] = uigetfile({'*.jpg;*.jpeg;*.png;*.bmp', 'Image Files'});
            if isequal(fileName, 0)
                return;
            end

            app.CurrentImagePath = string(fullfile(folderPath, fileName));
            app.CurrentImage = imread(app.CurrentImagePath);
            app.ImagePathField.Value = app.CurrentImagePath;
            imshow(app.CurrentImage, "Parent", app.InputAxes);
            title(app.InputAxes, "Loaded Image")
            app.appendStatus("Loaded image: " + app.CurrentImagePath);
        end

        function onRunPipeline(app)
            if isempty(app.CurrentImage)
                app.appendStatus("Load an image before running the pipeline.");
                return;
            end

            drawnow;
            app.appendStatus("Running pipeline...");
            app.LastResult = runLPRPipeline(app.CurrentImage, app.Config);
            app.renderResult(app.LastResult);
        end

        function onReset(app)
            app.CurrentImage = [];
            app.CurrentImagePath = "";
            app.LastResult = struct();
            app.resetView();
        end

        function resetView(app)
            cla(app.InputAxes);
            cla(app.PlateAxes);
            cla(app.DebugAxes);
            axis(app.InputAxes, "off")
            axis(app.PlateAxes, "off")
            axis(app.DebugAxes, "off")

            app.ImagePathField.Value = "";
            app.PlateTextField.Value = "";
            app.StateField.Value = "";
            app.CategoryField.Value = "";
            app.ConfidenceField.Value = "";
            app.StatusArea.Value = "Ready. Load an image to begin.";
        end

        function renderResult(app, result)
            if isfield(result.debug, "overlay") && ~isempty(result.debug.overlay)
                imshow(result.debug.overlay, "Parent", app.InputAxes);
            else
                imshow(app.CurrentImage, "Parent", app.InputAxes);
            end
            title(app.InputAxes, "Detection Overlay")

            if ~isempty(result.rectifiedPlate)
                imshow(result.rectifiedPlate, "Parent", app.PlateAxes);
                title(app.PlateAxes, "Rectified Plate")
            else
                cla(app.PlateAxes);
            end

            if isfield(result.debug, "ocrInputPlate") && ~isempty(result.debug.ocrInputPlate)
                imshow(result.debug.ocrInputPlate, "Parent", app.DebugAxes);
                title(app.DebugAxes, "OCR Input")
            else
                cla(app.DebugAxes);
            end

            app.PlateTextField.Value = char(result.recognizedText);
            app.StateField.Value = char(result.stateInfo.name);
            app.CategoryField.Value = char(result.stateInfo.category);
            app.ConfidenceField.Value = sprintf("%.2f", result.confidence);

            app.appendStatus("Status: " + string(result.status));
            if isfield(result, "recognitionPath")
                app.appendStatus("Recognition path: " + string(result.recognitionPath));
            end
            app.appendRecognitionStatus(result);
            for i = 1:numel(result.messages)
                app.appendStatus(result.messages(i));
            end
        end

        function appendStatus(app, message)
            message = string(message);
            existingMessages = string(app.StatusArea.Value);

            if isempty(existingMessages)
                app.StatusArea.Value = message;
            else
                app.StatusArea.Value = [existingMessages(:); message];
            end
        end

        function appendRecognitionStatus(app, result)
            if ~isfield(result, "debug") || ~isfield(result.debug, "recognition") || isempty(result.debug.recognition)
                return;
            end

            recognitionMeta = result.debug.recognition;
            method = "unknown";
            if isfield(recognitionMeta, "method") && strlength(string(recognitionMeta.method)) > 0
                method = string(recognitionMeta.method);
            end

            app.appendStatus("OCR method: " + method);

            if isfield(recognitionMeta, "matlabOcr") && isstruct(recognitionMeta.matlabOcr)
                if isfield(recognitionMeta.matlabOcr, "success")
                    app.appendStatus("MATLAB OCR success: " + string(logical(recognitionMeta.matlabOcr.success)));
                end
                if isfield(recognitionMeta.matlabOcr, "text") && strlength(string(recognitionMeta.matlabOcr.text)) > 0
                    app.appendStatus("MATLAB OCR raw text: " + string(recognitionMeta.matlabOcr.text));
                end
                if isfield(recognitionMeta.matlabOcr, "confidence")
                    app.appendStatus("MATLAB OCR confidence: " + sprintf("%.2f", double(recognitionMeta.matlabOcr.confidence)));
                end
                if isfield(recognitionMeta, "ocrInputName") && strlength(string(recognitionMeta.ocrInputName)) > 0
                    app.appendStatus("OCR input variant: " + string(recognitionMeta.ocrInputName));
                end
            end
        end
    end
end
