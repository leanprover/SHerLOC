"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<i1>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<complex<f32>>, tensor<complex<f32>>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<i1>
    %5 = "stablehlo.real"(%3#0) : (tensor<complex<f32>>) -> tensor<f32>
    %6 = "stablehlo.real"(%3#1) : (tensor<complex<f32>>) -> tensor<f32>
    %7 = "stablehlo.compare"(%5, %6) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %8 = "stablehlo.imag"(%3#0) : (tensor<complex<f32>>) -> tensor<f32>
    %9 = "stablehlo.imag"(%3#1) : (tensor<complex<f32>>) -> tensor<f32>
    %10 = "stablehlo.compare"(%8, %9) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LE>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %11 = "stablehlo.real"(%3#0) : (tensor<complex<f32>>) -> tensor<f32>
    %12 = "stablehlo.real"(%3#1) : (tensor<complex<f32>>) -> tensor<f32>
    %13 = "stablehlo.compare"(%11, %12) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LE>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %14 = "stablehlo.select"(%7, %10, %13) : (tensor<i1>, tensor<i1>, tensor<i1>) -> tensor<i1>
    "stablehlo.custom_call"(%14, %4) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<i1>, tensor<i1>) -> ()
    "func.return"(%14) : (tensor<i1>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<complex<f32>>, tensor<complex<f32>>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<(1.10811841,-6.992440e-02)> : tensor<complex<f32>>}> : () -> tensor<complex<f32>>
    %2 = "stablehlo.constant"() <{value = dense<(3.3754487,-2.81067419)> : tensor<complex<f32>>}> : () -> tensor<complex<f32>>
    "func.return"(%1, %2) : (tensor<complex<f32>>, tensor<complex<f32>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<i1>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<true> : tensor<i1>}> : () -> tensor<i1>
    "func.return"(%0) : (tensor<i1>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

