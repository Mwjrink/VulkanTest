#pragma once

#include <algorithm>
#include <ostream>
#include <string>

#define CHECK "Y"
#define CROSS "X"

// TODO: @MaxPlatformSpecific, find a way to timestamp in each os, add a function: Log(Error_Level(ENUM) e, message); and
// LogWithoutLineStampTimeStamp(message);, or add an optional boolean to choose no stamping
class Logger
{
  private:
    std::ostream* stream;
    int           maxWidth;

    static inline void ReplaceAll(std::string& str, const std::string& from, const std::string& to)
    {
        size_t start_pos = 0;
        while ((start_pos = str.find(from, start_pos)) != std::string::npos)
        {
            str.replace(start_pos, from.length(), to);
            start_pos += to.length();
        }
    }

    void NormalizeInput(std::string& input)
    {
        ReplaceAll(input, "\t", "    ");
        // TODO: @MaxCompleteLogger, make sure this adds a new line char recursively incase the input string is many many
        // times too long for the log, also make the new lines start with whitespace equal to the length of the stamp
        if (input.length() > maxWidth)
        {
            input.insert(input.begin() + maxWidth - 1, '\n');
        }
        //
    }

  public:
    Logger() {}

    Logger(std::ostream* stream, int maxWidth = 80) : stream(stream), maxWidth(maxWidth) {}

    void Log(std::string record) { *stream << record << std::endl; }

    void Log() { *stream << std::endl; }

    void Log(std::string elementOne, std::string elementTwo, int distance, char separator = ' ')
    {
        NormalizeInput(elementOne);
        NormalizeInput(elementTwo);

        if (distance < 0)
        {
            Log(elementOne + std::string((maxWidth + distance) - elementOne.length(), separator) + elementTwo);
        }
        else
        {
            Log(elementOne + std::string(maxWidth - elementOne.length(), separator) + elementTwo);
        }
    }

    void Log(std::string elementOne, int distance)
    {
        NormalizeInput(elementOne);

        if (distance < 0)
            Log(std::string((maxWidth - distance) - elementOne.length(), ' ') + elementOne);
        else
            Log(std::string(distance, ' ') + elementOne);
    }

    void LogHeader(std::string header)
    {
        NormalizeInput(header);
        auto border = std::string(maxWidth, '=');
        Log(border);
        Log(header);
        Log(border);
        Log();
    }

    void LogMaxWidth(char c) { Log(std::string(maxWidth, c)); }

    void Flush() { stream->flush(); }
};