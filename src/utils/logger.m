function logger(level, message, varargin)
%LOGGER Lightweight console logger for the project.

if nargin < 2
    return;
end

timestamp = datestr(now, "yyyy-mm-dd HH:MM:SS");
formatted = sprintf(char(string(message)), varargin{:});
fprintf("[%s] [%s] %s\n", timestamp, upper(string(level)), formatted);
end
