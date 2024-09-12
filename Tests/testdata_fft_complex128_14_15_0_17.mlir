"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<14x15x0x33xf64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<14x15x0x17xcomplex<f64>>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<14x15x0x33xf64>
    %4 = "stablehlo.fft"(%2) <{fft_length = array<i64: 33>, fft_type = #stablehlo<fft_type IRFFT>}> : (tensor<14x15x0x17xcomplex<f64>>) -> tensor<14x15x0x33xf64>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<14x15x0x33xf64>, tensor<14x15x0x33xf64>) -> ()
    "func.return"(%4) : (tensor<14x15x0x33xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<14x15x0x17xcomplex<f64>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<> : tensor<14x15x0x17xcomplex<f64>>}> : () -> tensor<14x15x0x17xcomplex<f64>>
    "func.return"(%1) : (tensor<14x15x0x17xcomplex<f64>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<14x15x0x33xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<> : tensor<14x15x0x33xf64>}> : () -> tensor<14x15x0x33xf64>
    "func.return"(%0) : (tensor<14x15x0x33xf64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

