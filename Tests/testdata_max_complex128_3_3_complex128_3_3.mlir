"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3x3xcomplex<f64>>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<3x3xcomplex<f64>>, tensor<3x3xcomplex<f64>>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<3x3xcomplex<f64>>
    %5 = "stablehlo.real"(%3#0) : (tensor<3x3xcomplex<f64>>) -> tensor<3x3xf64>
    %6 = "stablehlo.real"(%3#1) : (tensor<3x3xcomplex<f64>>) -> tensor<3x3xf64>
    %7 = "stablehlo.compare"(%5, %6) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xi1>
    %8 = "stablehlo.compare"(%5, %6) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GT>}> : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xi1>
    %9 = "stablehlo.imag"(%3#0) : (tensor<3x3xcomplex<f64>>) -> tensor<3x3xf64>
    %10 = "stablehlo.imag"(%3#1) : (tensor<3x3xcomplex<f64>>) -> tensor<3x3xf64>
    %11 = "stablehlo.compare"(%9, %10) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GT>}> : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xi1>
    %12 = "stablehlo.select"(%7, %11, %8) : (tensor<3x3xi1>, tensor<3x3xi1>, tensor<3x3xi1>) -> tensor<3x3xi1>
    %13 = "stablehlo.select"(%12, %3#0, %3#1) : (tensor<3x3xi1>, tensor<3x3xcomplex<f64>>, tensor<3x3xcomplex<f64>>) -> tensor<3x3xcomplex<f64>>
    "stablehlo.custom_call"(%13, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<3x3xcomplex<f64>>, tensor<3x3xcomplex<f64>>) -> ()
    "func.return"(%13) : (tensor<3x3xcomplex<f64>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<3x3xcomplex<f64>>, tensor<3x3xcomplex<f64>>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[(0x7FF8000000000000,0.000000e+00), (0x7FF8000000000000,0.000000e+00), (0x7FF8000000000000,0.000000e+00)], [(0x7FF0000000000000,0.000000e+00), (0x7FF0000000000000,0.000000e+00), (0x7FF0000000000000,0.000000e+00)], [(0xFFF0000000000000,0.000000e+00), (0xFFF0000000000000,0.000000e+00), (0xFFF0000000000000,0.000000e+00)]]> : tensor<3x3xcomplex<f64>>}> : () -> tensor<3x3xcomplex<f64>>
    %2 = "stablehlo.constant"() <{value = dense<[[(0x7FF8000000000000,0.000000e+00), (0x7FF0000000000000,0.000000e+00), (0xFFF0000000000000,0.000000e+00)], [(0x7FF8000000000000,0.000000e+00), (0x7FF0000000000000,0.000000e+00), (0xFFF0000000000000,0.000000e+00)], [(0x7FF8000000000000,0.000000e+00), (0x7FF0000000000000,0.000000e+00), (0xFFF0000000000000,0.000000e+00)]]> : tensor<3x3xcomplex<f64>>}> : () -> tensor<3x3xcomplex<f64>>
    "func.return"(%1, %2) : (tensor<3x3xcomplex<f64>>, tensor<3x3xcomplex<f64>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x3xcomplex<f64>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[(0x7FF8000000000000,0.000000e+00), (0x7FF0000000000000,0.000000e+00), (0xFFF0000000000000,0.000000e+00)], [(0x7FF8000000000000,0.000000e+00), (0x7FF0000000000000,0.000000e+00), (0x7FF0000000000000,0.000000e+00)], [(0x7FF8000000000000,0.000000e+00), (0x7FF0000000000000,0.000000e+00), (0xFFF0000000000000,0.000000e+00)]]> : tensor<3x3xcomplex<f64>>}> : () -> tensor<3x3xcomplex<f64>>
    "func.return"(%0) : (tensor<3x3xcomplex<f64>>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

