#pragma once

#include <time.h>
#include <algorithm>
#include <ctime>
#include <iomanip>
#include <ostream>
#include <string>

#define CHECK "Y"
#define CROSS "X"

// TODO: @MaxPlatformSpecific, find a way to timestamp in each os, add a function: Log(Error_Level(ENUM) e, message); and
// LogWithoutLineStampTimeStamp(message);, or add an optional boolean to choose no stamping
class Logger
{
  private:
    const int     TIMESTAMP_LENGTH = 10;
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

    void Log(std::string record)
    {
        // auto t  = std::time(nullptr);
        // auto tm = *std::localtime(&t);
        *stream <<
            //"[" << std::put_time(&tm, "%H-%M-%S") << "] " <<
            record << std::endl;
    }

    void LogWithoutTimestamp(std::string record) { *stream << record << std::endl; }

    void Log() { *stream << std::endl; }

    void Log(std::string elementOne, std::string elementTwo, int distance, char separator = ' ', bool timestamp = true)
    {
        NormalizeInput(elementOne);
        NormalizeInput(elementTwo);

        if (timestamp)
        {
            if (distance < 0)
            {
                Log(elementOne + std::string(((size_t)maxWidth + distance) - elementOne.length(), separator) + elementTwo);
            }
            else
            {
                Log(elementOne + std::string(maxWidth - elementOne.length(), separator) + elementTwo);
            }
        }
        else
        {
            if (distance < 0)
            {
                LogWithoutTimestamp(
                    elementOne +
                    std::string(((size_t)maxWidth + TIMESTAMP_LENGTH + distance) - elementOne.length(), separator) +
                    elementTwo);
            }
            else
            {
                LogWithoutTimestamp(elementOne +
                                    std::string((size_t)maxWidth + TIMESTAMP_LENGTH - elementOne.length(), separator) +
                                    elementTwo);
            }
        }
    }

    void Log(std::string elementOne, int distance, bool timestamp = true)
    {
        NormalizeInput(elementOne);

        if (timestamp)
        {
            if (distance < 0)
                Log(std::string(((size_t)maxWidth - distance) - elementOne.length(), ' ') + elementOne);
            else
                Log(std::string(distance, ' ') + elementOne);
        }
        else
        {
            if (distance < 0)
                LogWithoutTimestamp(
                    std::string(((size_t)maxWidth + TIMESTAMP_LENGTH - distance) - elementOne.length(), ' ') + elementOne);
            else
                LogWithoutTimestamp(std::string((size_t)distance + TIMESTAMP_LENGTH, ' ') + elementOne);
        }
    }

    void LogHeader(std::string header, bool timestamp = false)
    {
        NormalizeInput(header);
        if (timestamp)
        {
            auto border = std::string((size_t)maxWidth + TIMESTAMP_LENGTH, '=');
            LogWithoutTimestamp(border);
            LogWithoutTimestamp(header);
            LogWithoutTimestamp(border);
        }
        else
        {
            auto border = std::string(maxWidth, '=');
            Log(border);
            Log(header);
            Log(border);
        }
        Log();
    }

    void LogMaxWidth(char c, bool timestamp = true)
    {
        if (timestamp)
            Log(std::string(maxWidth, c));
        else
            LogWithoutTimestamp(std::string((size_t)maxWidth + TIMESTAMP_LENGTH, c));
    }

    void Flush() { stream->flush(); }
};