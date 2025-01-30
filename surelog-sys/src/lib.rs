use std::ffi::{CStr, CString};

pub mod bindings;

pub struct VpiIterator {
    iter: bindings::vpiHandle,
}

// TODO: Figure out why time is stored in two separate integers
pub enum VpiTime {
    SimTime { high: u32, low: u32 },
    ScaledRealTime(f64),
    SuppressTime,
}

impl VpiTime {
    fn from_vpi_time(time: bindings::t_vpi_time) -> Self {
        match time.type_ as u32 {
            bindings::vpiSimTime => VpiTime::SimTime {
                high: time.high,
                low: time.low,
            },
            bindings::vpiScaledRealTime => VpiTime::ScaledRealTime(time.real),
            bindings::vpiSuppressTime => VpiTime::SuppressTime,
            _ => unreachable!(),
        }
    }
}

pub enum VpiValue {
    BinaryString(String),
    OctalString(String),
    DecimalString(String),
    HexadecimalString(String),
    IntValue(i64),
    UintValue(u64),
    RealValue(f64),
    StringValue(String),
    TimeValue(VpiTime),
    ObjectTypeValue,
}

impl Iterator for VpiIterator {
    type Item = VpiHandle;

    fn next(&mut self) -> Option<Self::Item> {
        unsafe {
            let next = bindings::vpi_scan(self.iter);

            if next.is_null() {
                None
            } else {
                Some(VpiHandle::new(next))
            }
        }
    }
}

unsafe fn cstr_to_string(value: *mut i8) -> String {
    CStr::from_ptr(value).to_str().unwrap().to_owned()
}

// Wrapper for VpiHandles
#[derive(Copy, Clone)]
pub struct VpiHandle {
    handle: bindings::vpiHandle,
}

impl VpiHandle {
    pub(crate) fn new(handle: bindings::vpiHandle) -> Self {
        Self { handle }
    }

    pub fn vpi_get(&self, param: u32) -> u32 {
        unsafe { bindings::vpi_get(param as i32, self.handle) as u32 }
    }

    pub fn vpi_get_bool(&self, param: u32) -> bool {
        let value = self.vpi_get(param);

        // TODO: Check if this is actually correct.
        return value == 1;
    }

    pub fn vpi_handle(&self, param: u32) -> Option<VpiHandle> {
        unsafe {
            let handle = bindings::vpi_handle(param as i32, self.handle);

            if handle.is_null() {
                None
            } else {
                Some(Self { handle })
            }
        }
    }

    pub fn vpi_iter(&self, param: u32) -> VpiIterator {
        let iter = unsafe { bindings::vpi_iterate(param as i32, self.handle) };
        VpiIterator { iter }
    }

    pub fn vpi_str(&self, param: u32) -> String {
        unsafe {
            let value = bindings::vpi_get_str(param as i32, self.handle);

            if value.is_null() {
                // NOTE: Maybe not the best idea but I think that's how they do it
                "".to_owned()
            } else {
                cstr_to_string(value)
            }
        }
    }

    pub fn vpi_get_value(&self) -> VpiValue {
        // Initialize to a sane default
        let mut vpi_val = bindings::t_vpi_value {
            format: 0,
            value: bindings::t_vpi_value__bindgen_ty_1 { integer: 0 },
        };

        // Load the value inside vpi_val
        unsafe {
            bindings::vpi_get_value(self.handle, &mut vpi_val);

            // Move the union to a proper rust enum
            match vpi_val.format as u32 {
                bindings::vpiBinStrVal => {
                    VpiValue::BinaryString(cstr_to_string(vpi_val.value.str_))
                }
                bindings::vpiOctStrVal => VpiValue::OctalString(cstr_to_string(vpi_val.value.str_)),
                bindings::vpiDecStrVal => {
                    VpiValue::DecimalString(cstr_to_string(vpi_val.value.str_))
                }
                bindings::vpiHexStrVal => {
                    VpiValue::HexadecimalString(cstr_to_string(vpi_val.value.str_))
                }
                bindings::vpiStringVal => VpiValue::StringValue(cstr_to_string(vpi_val.value.str_)),

                bindings::vpiIntVal => VpiValue::IntValue(vpi_val.value.integer),
                bindings::vpiUIntVal => VpiValue::UintValue(vpi_val.value.uint),
                bindings::vpiRealVal => VpiValue::RealValue(vpi_val.value.real),
                bindings::vpiTimeVal => {
                    VpiValue::TimeValue(VpiTime::from_vpi_time(*vpi_val.value.time))
                }

                bindings::vpiScalarVal => todo!("constant vpiScalarVal"),
                bindings::vpiVectorVal => todo!("constant vpiVectorVal"),
                bindings::vpiStrengthVal => todo!("constant vpiStrengthVal"),
                bindings::vpiObjTypeVal => todo!("constant vpiObjTypeVal"),

                // FIXME: We should probably handle this better
                _ => todo!("unsupported constant format {}", vpi_val.format),
            }
        }
    }

    pub fn name(&self) -> String {
        self.vpi_str(bindings::vpiName)
    }

    pub fn vpi_type(&self) -> u32 {
        self.vpi_get(bindings::vpiType) as u32
    }

    pub fn vpi_debug_print(&self) {
        unsafe {
            bindings::vpi_visit(self.handle);
        }
    }
}

pub struct SVDesign {
    design: *mut bindings::SystemVerilogDesign,
    handle: bindings::vpiHandle,
}

impl SVDesign {
    pub fn get_top(&self) -> Option<VpiHandle> {
        let top = unsafe {
            let top = bindings::design_top_entity(self.handle);
            if top.is_null() {
                None
            } else {
                Some(top)
            }
        };

        top.map(VpiHandle::new)
    }

    pub fn get_name(&self) -> String {
        unsafe {
            let top = bindings::design_top_entity(self.handle);

            let name_raw = bindings::vpi_get_str(bindings::vpiName as i32, top);
            let name = CStr::from_ptr(name_raw);

            name.to_str().unwrap().to_owned()
        }
    }
}

impl Drop for SVDesign {
    fn drop(&mut self) {
        unsafe {
            bindings::design_free(self.design);
        }
    }
}

pub fn compile(sources: &[&str]) -> SVDesign {
    let c_string_container = sources
        .iter()
        .map(|x| CString::new(*x).unwrap())
        .collect::<Vec<_>>();

    let c_strings = c_string_container
        .iter()
        .map(|x| x.as_ptr())
        .collect::<Vec<_>>();

    unsafe {
        let design = bindings::design_create();
        let handle = bindings::design_compile(design, c_strings.as_ptr(), c_strings.len());

        SVDesign { design, handle }
    }
}
