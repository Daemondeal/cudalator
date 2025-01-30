#include "CppEmitter.hpp"
#include <algorithm>
#include <fmt/format.h>

namespace cudalator {

CppEmitter::CppEmitter(std::ostream& out) : m_out(out), m_indent(0) {}

void CppEmitter::emitInclude(std::string_view include) {
    emitIndent();
    emit("#include \"");
    emit(include);
    emitLine("\"");
}

void CppEmitter::emitNamespaceStart(std::string_view name) {
    emitIndent();
    emit("namespace ");
    emit(name);
    emitLine(" {");
}
void CppEmitter::emitNamespaceEnd(std::string_view name) {
    emit("} // ");
    emitLine(name);
}

void CppEmitter::emit(std::string_view string) {
    m_out << string;
}

void CppEmitter::emitLine(std::string_view string) {
    m_out << string << std::endl;
}

void CppEmitter::emitLine() {
    m_out << std::endl;
}

void CppEmitter::emitName(std::string_view name) {
    m_out << cleanName(name);
}

// FIXME: This is not how we should do it, find a better way. Maybe using $?
std::string CppEmitter::cleanName(std::string_view name) {
    std::string clone(name);
    std::replace(clone.begin(), clone.end(), '@', '_');
    std::replace(clone.begin(), clone.end(), '.', '_');

    return clone;
}

void CppEmitter::emitStructStart(std::string_view name) {
    emitIndent();

    m_out << "struct " << cleanName(name) << " {\n";
    m_indent++;
}

void CppEmitter::emitStructEnd(std::string_view name) {
    m_indent--;
    emitIndent();
    m_out << "}; // struct " << cleanName(name) << "\n";
}

void CppEmitter::emitBlockStart() {
    m_out << "{\n";
    m_indent++;
}
void CppEmitter::emitBlockEnd() {
    m_indent--;
    emitIndent();
    m_out << "}\n";
}

void CppEmitter::emitIndent() {
    for (size_t i = 0; i < m_indent; i++) {
        m_out << " ";
    }
}
} // namespace cudalator
