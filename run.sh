#! /bin/bash

ir_converter_main --top=add_two_f32 adler32.x > adler32.ir
opt_main adler32.ir > adler32.opt.ir
codegen_main --pipeline_stages=1 --delay_model=unit adler32.opt.ir > adler32.sv