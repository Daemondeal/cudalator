#pragma once

#include <ostream>
namespace cudalator {

class CppEmitter {
public:
    CppEmitter() = delete;
    CppEmitter(std::ostream& out);

    void emitInclude(std::string_view include);

    void emitNamespaceStart(std::string_view name);
    void emitNamespaceEnd(std::string_view name);

    void emitStart();
    void emitEnd();

    void emit(std::string_view string);
    void emitLine(std::string_view string);
    void emitLine();

    void emitName(std::string_view name);

    void emitStructStart(std::string_view name);
    void emitStructEnd(std::string_view name);

    void emitBlockStart();
    void emitBlockEnd();

    void emitIndent();

private:
    std::string cleanName(std::string_view name);

    std::ostream& m_out;

    size_t m_indent;
};

} // namespace cudalator
