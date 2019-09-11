#pragma once

#include <ostream>
#include <string>

#define CHECK "Y"
#define CROSS "X"

// TODO: @MaxPlatformSpecific, find a way to timestamp in each os, add a function: Log(Error_Level(ENUM) e, message); and LogWithoutLineStampTimeStamp(message);
class Logger
{
    std::ostream* stream;
    int           maxWidth;

  public:
    Logger() {}

    Logger(std::ostream* stream, int maxWidth = 80) : stream(stream), maxWidth(maxWidth) {}

    void Log(std::string record) { *stream << record << std::endl; }

    void Log() { *stream << std::endl; }

    void Log(std::string elementOne, std::string elementTwo, int distance, char separator)
    {
        if (distance < 0)
        {
            Log(elementOne + std::string((maxWidth + distance) - elementOne.length(), separator) + elementTwo);
        }
        else
        {
            Log(elementOne + std::string(maxWidth - elementOne.length(), separator) + elementTwo);
        }
    }

    void Log(std::string elementOne, std::string elementTwo, int distance) { Log(elementOne, elementTwo, distance, ' '); }

    void Log(std::string elementOne, int distance)
    {
        if (distance < 0)
            Log(std::string((maxWidth - distance) - elementOne.length(), ' ') + elementOne);
        else
            Log(std::string(distance, ' ') + elementOne);
    }

    void LogHeader(std::string header)
    {
        auto border = std::string(maxWidth, '=');
        Log(border);
        Log(header);
        Log(border);
        Log();
    }

    void LogMaxWidth(char c) { Log(std::string(maxWidth, c)); }

    void Flush() { stream->flush(); }
};