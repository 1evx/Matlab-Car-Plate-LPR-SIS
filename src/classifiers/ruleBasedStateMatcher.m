function stateInfo = ruleBasedStateMatcher(plateText, rules)
%RULEBASEDSTATEMATCHER Match Malaysian state/category rules against plate text.

normalized = upper(regexprep(string(plateText), "[^A-Z0-9]", ""));
stateInfo = struct( ...
    "name", "Unknown", ...
    "category", "Unknown", ...
    "matched", false, ...
    "matchedRule", "", ...
    "description", "No matching Malaysian registration rule found.");

if strlength(normalized) == 0
    return;
end

for i = 1:numel(rules)
    rule = rules(i);
    isMatch = false;

    switch string(rule.matcherType)
        case "prefix"
            isMatch = startsWith(normalized, string(rule.token));
        case "pattern"
            isMatch = ~isempty(regexp(normalized, rule.token, "once"));
    end

    if isMatch
        stateInfo = struct( ...
            "name", string(rule.name), ...
            "category", string(rule.category), ...
            "matched", true, ...
            "matchedRule", string(rule.token), ...
            "description", string(rule.description));
        return;
    end
end
end

