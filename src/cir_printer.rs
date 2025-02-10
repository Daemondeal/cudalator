use std::io::Write;

use crate::cir::{
    Ast, ConstantIdx, ConstantKind, ExprIdx, ExprKind, ModuleIdx, PortDirection, ProcessIdx,
    ScopeIdx, SelectKind, SensitivtyKind, Signal, SignalIdx, SignalLifetime, StatementIdx,
    StatementKind, TypeIdx, TypeKind,
};

type Res = Result<(), std::io::Error>;

// FIXME: This module is written this way just to make everything easier to write and to have
// something working quickly, but ideally there wouldn't be this many unnecessary allocations.
// Since this is only a module intended for debugging reasons it doesn't matter too much, but maybe
// in the future we could make this better.
pub fn print_cir_ast<W: Write>(ast: &Ast, writer: &mut W) -> Res {
    let sexpr = cir_to_sexpr(ast);

    sexpr.print(writer, 0)?;
    write!(writer, "\n")
}

fn cir_to_sexpr<'a>(ast: &'a Ast) -> SExpr<'a> {
    let top_module = ast.top_module.expect("No top module");

    sexpr_convert_module(ast, top_module)
}

fn sexpr_convert_module<'a>(ast: &'a Ast, module_idx: ModuleIdx) -> SExpr<'a> {
    let module = ast.get_module(module_idx);

    let mut components = vec![];

    let mut ports = vec![];
    for port in &module.ports {
        let direction = match port.direction {
            PortDirection::Input => "input",
            PortDirection::Output => "output",
            PortDirection::Inout => "inout",
            PortDirection::Invalid => "invalid",
        };

        let signal_name = &ast.get_signal(port.signal).token.name;

        ports.push(SExpr::expr(signal_name, vec![SExpr::atom(direction)]));
    }
    components.push(SExpr::statement("ports", ports));

    components.push(sexpr_convert_scope(ast, module.scope));

    for process in &module.processes {
        components.push(sexpr_convert_process(ast, *process));
    }

    SExpr::Statement {
        name: &module.token.name,
        children: components,
    }
}

fn sexpr_convert_process<'a>(ast: &'a Ast, process_idx: ProcessIdx) -> SExpr<'a> {
    let process = ast.get_process(process_idx);
    let mut children = vec![];

    let mut sensitivity = vec![];
    for (kind, signal) in &process.sensitivity_list {
        let signal_atom = SExpr::atom(&ast.get_signal(*signal).token.name);

        match kind {
            SensitivtyKind::OnChange => sensitivity.push(signal_atom),
            SensitivtyKind::Posedge => sensitivity.push(SExpr::expr("posedge", vec![signal_atom])),
            SensitivtyKind::Negedge => sensitivity.push(SExpr::expr("negedge", vec![signal_atom])),
        };
    }
    if sensitivity.is_empty() {
        children.push(SExpr::atom("(sensitivity_empty)"))
    } else {
        children.push(SExpr::expr("sensitivity", sensitivity));
    }

    children.push(sexpr_convert_scope(ast, process.scope));

    for statement in &process.statements {
        children.push(sexpr_convert_statement(ast, *statement));
    }

    SExpr::statement("process", children)
}

fn sexpr_convert_statement<'a>(ast: &'a Ast, statement_idx: StatementIdx) -> SExpr<'a> {
    let statement = ast.get_statement(statement_idx);

    match &statement.kind {
        StatementKind::Invalid => SExpr::atom("invalid_stmt"),
        StatementKind::Assignment { lhs, rhs, select } => {
            let signal = sexpr_convert_signal(ast, *lhs);

            let target = match select {
                SelectKind::None => signal,
                SelectKind::Bit(expr_idx) => SExpr::expr(
                    "select_bit",
                    vec![sexpr_convert_expr(ast, *expr_idx), signal],
                ),
                SelectKind::Parts { lhs, rhs } => SExpr::expr(
                    "select_part",
                    vec![
                        sexpr_convert_expr(ast, *lhs),
                        sexpr_convert_expr(ast, *rhs),
                        signal,
                    ],
                ),
            };

            SExpr::expr("assign", vec![target, sexpr_convert_expr(ast, *rhs)])
        }

        StatementKind::Block { statements, scope } => SExpr::statement(
            "block",
            vec![
                sexpr_convert_scope(ast, *scope),
                SExpr::Statement {
                    name: "statements",
                    children: statements
                        .iter()
                        .map(|s| sexpr_convert_statement(ast, *s))
                        .collect(),
                },
            ],
        ),

        StatementKind::If {
            condition,
            body,
            else_,
        } => {
            let mut statements = vec![
                SExpr::expr("condition", vec![sexpr_convert_expr(ast, *condition)]),
                sexpr_convert_statement(ast, *body),
            ];
            if let Some(else_) = else_ {
                statements.push(sexpr_convert_statement(ast, *else_));
            }

            SExpr::statement("if", statements)
        }
        StatementKind::While { condition, body } => SExpr::statement(
            "while",
            vec![
                SExpr::expr("condition", vec![sexpr_convert_expr(ast, *condition)]),
                sexpr_convert_statement(ast, *body),
            ],
        ),

        StatementKind::DoWhile { condition, body } => SExpr::statement(
            "do_while",
            vec![
                SExpr::expr("condition", vec![sexpr_convert_expr(ast, *condition)]),
                sexpr_convert_statement(ast, *body),
            ],
        ),

        StatementKind::Case => todo!(),
        StatementKind::Foreach => todo!(),

        StatementKind::Return { expr } => {
            SExpr::expr("return", vec![sexpr_convert_expr(ast, *expr)])
        }
        StatementKind::Break => SExpr::atom("break"),
        StatementKind::Continue => SExpr::atom("continue"),
        StatementKind::Null => SExpr::atom("null"),
    }
}

fn sexpr_convert_expr<'a>(ast: &'a Ast, expr_idx: ExprIdx) -> SExpr<'a> {
    let expr = ast.get_expr(expr_idx);

    match &expr.kind {
        ExprKind::Invalid => SExpr::atom("invalid_expr"),
        ExprKind::Constant { constant } => sexpr_convert_constant(ast, *constant),
        ExprKind::Unary { op, expr } => {
            SExpr::expr(op.to_str(), vec![sexpr_convert_expr(ast, *expr)])
        }
        ExprKind::Binary { op, lhs, rhs } => SExpr::expr(
            op.to_str(),
            vec![sexpr_convert_expr(ast, *lhs), sexpr_convert_expr(ast, *rhs)],
        ),
        ExprKind::Concatenation { exprs } => SExpr::expr(
            "concat",
            exprs.iter().map(|e| sexpr_convert_expr(ast, *e)).collect(),
        ),
        ExprKind::PartSelect { lhs, rhs, target } => SExpr::expr(
            "part_select",
            vec![
                sexpr_convert_expr(ast, *lhs),
                sexpr_convert_expr(ast, *rhs),
                sexpr_convert_signal(ast, *target),
            ],
        ),
        ExprKind::BitSelect { expr, target } => SExpr::expr(
            "bit_select",
            vec![
                sexpr_convert_expr(ast, *expr),
                sexpr_convert_signal(ast, *target),
            ],
        ),
        ExprKind::SignalRef { signal } => sexpr_convert_signal(ast, *signal),
    }
}

// TODO: Maybe this should remove also the top entity's name
fn clean_signal_name(signal: &Signal) -> &str {
    let signal_name = &signal.full_name;
    let split_name = signal_name.split("@").collect::<Vec<_>>();
    if split_name.len() > 1 {
        split_name[1]
    } else {
        signal_name
    }
}

fn sexpr_convert_signal<'a>(ast: &'a Ast, signal_idx: SignalIdx) -> SExpr<'a> {
    let signal = ast.get_signal(signal_idx);
    let name = clean_signal_name(signal);

    SExpr::atom(name)
}

fn sexpr_convert_constant<'a>(ast: &'a Ast, constant_idx: ConstantIdx) -> SExpr<'a> {
    let constant = ast.get_constant(constant_idx);

    // FIXME: THIS FUNCTION IS VERY BAD. We are leaking memory because we don't
    //        have a good place to put this temporary string.
    //        Find a way to manage this better.

    match &constant.kind {
        ConstantKind::Integer(val) => {
            let value = format!("{}", val).leak();
            SExpr::expr("constant_int", vec![SExpr::atom(value)])
        }

        ConstantKind::UnsignedInteger(val) => {
            let value = format!("{}", val).leak();
            SExpr::expr("constant_uint", vec![SExpr::atom(value)])
        }

        ConstantKind::Value { vals } => {
            let children = vals
                .iter()
                .map(|v| {
                    let val_string = format!("{}", v).leak();
                    SExpr::atom(val_string)
                })
                .collect();
            SExpr::expr("constant", children)
        }
        ConstantKind::Invalid => SExpr::atom("invalid_constant"),
    }
}

fn sexpr_convert_scope<'a>(ast: &'a Ast, scope_idx: ScopeIdx) -> SExpr<'a> {
    let scope = ast.get_scope(scope_idx);

    if scope.signals.len() == 0 {
        return SExpr::atom("(empty scope)");
    }

    let mut children = vec![];

    for signal_idx in &scope.signals {
        let signal = ast.get_signal(*signal_idx);

        let signal_name = clean_signal_name(signal);
        let signal_type = sexpr_convert_type(ast, signal.typ);
        let signal_lifetime = match signal.lifetime {
            SignalLifetime::Static => SExpr::atom("var_static"),
            SignalLifetime::Automatic => SExpr::atom("var_automatic"),
            SignalLifetime::Net => SExpr::atom("net"),
        };

        children.push(SExpr::expr(signal_name, vec![signal_type, signal_lifetime]));
    }

    SExpr::statement("scope", children)
}

fn sexpr_convert_type<'a>(ast: &'a Ast, type_idx: TypeIdx) -> SExpr<'a> {
    let typ = ast.get_typ(type_idx);

    // FIXME: Fix the leak issue here as well
    match &typ.kind {
        TypeKind::Invalid => SExpr::atom("invalid_type"),
        TypeKind::Integer => SExpr::atom("integer"),
        TypeKind::Int => SExpr::atom("int"),
        TypeKind::Bit(range) => match range {
            Some(range) => SExpr::expr(
                "bit",
                vec![
                    SExpr::atom(range.left.to_string().leak()),
                    SExpr::atom(range.right.to_string().leak()),
                ],
            ),
            None => SExpr::atom("bit"),
        },
        TypeKind::Logic(range) => match range {
            Some(range) => SExpr::expr(
                "logic",
                vec![
                    SExpr::atom(range.left.to_string().leak()),
                    SExpr::atom(range.right.to_string().leak()),
                ],
            ),
            None => SExpr::atom("logic"),
        },
    }
}

enum SExpr<'a> {
    Statement {
        name: &'a str,
        children: Vec<SExpr<'a>>,
    },
    Expr {
        name: &'a str,
        children: Vec<SExpr<'a>>,
    },
    Atom {
        name: &'a str,
    },
}

impl<'a> SExpr<'a> {
    fn atom(name: &'a str) -> Self {
        Self::Atom { name }
    }

    fn expr(name: &'a str, children: Vec<SExpr<'a>>) -> Self {
        Self::Expr { name, children }
    }

    fn statement(name: &'a str, children: Vec<SExpr<'a>>) -> Self {
        Self::Statement { name, children }
    }

    fn print_indent<W: Write>(&self, writer: &mut W, indent: i32) -> Res {
        for _ in 0..indent {
            write!(writer, "  ")?;
        }

        Ok(())
    }
    fn print<W: Write>(&self, writer: &mut W, indent: i32) -> Res {
        match self {
            SExpr::Statement { name, children } => {
                writeln!(writer, "({name}")?;
                for child in children {
                    self.print_indent(writer, indent + 1)?;
                    child.print(writer, indent + 1)?;
                    write!(writer, "\n")?;
                }
                self.print_indent(writer, indent)?;
                write!(writer, ") // {name}")
            }

            SExpr::Expr { name, children } => {
                write!(writer, "({name}")?;
                for child in children {
                    write!(writer, " ")?;
                    child.print(writer, indent)?;
                }
                write!(writer, ")")
            }
            SExpr::Atom { name } => write!(writer, "{name}"),
        }
    }
}
