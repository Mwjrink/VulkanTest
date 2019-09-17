#pragma once

#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "Utilities.h"

// TODO: @MaxCompleteAPI, make a simple configurations class that stores configs in a .ini or some such file (maybe allow
//			it to be encrypted) and you can update those values and read them so they persist accross runs (for settings
//			and stuff)
class Configurations
{
  private:
    std::string                        _filename;
    std::map<std::string, std::string> _sconfigurations;
    std::map<std::string, float>       _fconfigurations;
    std::map<std::string, int>         _iconfigurations;
    std::fstream                       _stream;
    bool                               written = false;

    void flush()
    {
        auto fileContents = std::vector<char>();

        for (auto kvpair : _sconfigurations)
        {
            //
        }

        for (auto kvpair : _fconfigurations)
        {
            //
        }

        for (auto kvpair : _iconfigurations)
        {
            //
        }

        _stream.clear();
        _stream.seekp(0);
        _stream.write(fileContents.data(), fileContents.size());
        _stream.flush();

        written = true;
    }

  public:
    Configurations(std::string filename) : _filename(filename)
    {
        _stream = std::fstream(filename, std::ios::ate | std::ios::binary);

        // TODO: @MaxCompleteAPI, add support for creating the directory if it doesn't exist

        if (!_stream.is_open())
        {
            // TODO: @MaxCompleteAPI, log the errors produced here
            throw std::runtime_error("failed to open file!");
        }

        size_t            fileSize = (size_t)_stream.tellg();
        std::vector<char> buffer(fileSize);

        // seek back to the beginning of the file
        _stream.seekg(0);
        // read all of the bytes at once
        _stream.read(buffer.data(), fileSize);
    }

    ~Configurations()
    {
        if (!written)
        {
            flush();
        }
        else
        {
            _stream.flush();
        }

        _stream.close();
    }

    void set(std::string category, std::string key, std::string value) { _sconfigurations[category + "." + key] = value; }

    void set(std::string category, std::string key, float value) { _fconfigurations[category + "." + key] = value; }

    void set(std::string category, std::string key, int value) { _iconfigurations[category + "." + key] = value; }

    // void* get(std::string category, std::string key)
    //{
    //    if (_sconfigurations.find(category + "." + key) != _sconfigurations.end())
    //    {
    //        return &_sconfigurations[category + "." + key];
    //    }
    //    else if (_fconfigurations.find(category + "." + key) != _fconfigurations.end())
    //    {
    //        return &_fconfigurations[category + "." + key];
    //    }
    //    else if (_iconfigurations.find(category + "." + key) != _iconfigurations.end())
    //    {
    //        return &_iconfigurations[category + "." + key];
    //    }
    //}

    int geti(std::string category, std::string key)
    {
        if (_iconfigurations.find(category + "." + key) != _iconfigurations.end())
        {
            return _iconfigurations[category + "." + key];
        }
    }

    float getf(std::string category, std::string key)
    {
        if (_fconfigurations.find(category + "." + key) != _fconfigurations.end())
        {
            return _fconfigurations[category + "." + key];
        }
    }

    std::string gets(std::string category, std::string key)
    {
        if (_sconfigurations.find(category + "." + key) != _sconfigurations.end())
        {
            return _sconfigurations[category + "." + key];
        }
    }
};