"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3xcomplex<f32>>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<2x3xcomplex<f32>>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<3xcomplex<f32>>
    %4 = "stablehlo.constant"() <{value = dense<(0x7F800000,0.000000e+00)> : tensor<complex<f32>>}> : () -> tensor<complex<f32>>
    %5 = "stablehlo.reduce"(%2, %4) <{dimensions = array<i64: 0>}> ({
    ^bb0(%arg0: tensor<complex<f32>>, %arg1: tensor<complex<f32>>):
      %6 = "stablehlo.real"(%arg0) : (tensor<complex<f32>>) -> tensor<f32>
      %7 = "stablehlo.real"(%arg1) : (tensor<complex<f32>>) -> tensor<f32>
      %8 = "stablehlo.compare"(%6, %7) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %9 = "stablehlo.compare"(%6, %7) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %10 = "stablehlo.imag"(%arg0) : (tensor<complex<f32>>) -> tensor<f32>
      %11 = "stablehlo.imag"(%arg1) : (tensor<complex<f32>>) -> tensor<f32>
      %12 = "stablehlo.compare"(%10, %11) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %13 = "stablehlo.select"(%8, %12, %9) : (tensor<i1>, tensor<i1>, tensor<i1>) -> tensor<i1>
      %14 = "stablehlo.select"(%13, %arg0, %arg1) : (tensor<i1>, tensor<complex<f32>>, tensor<complex<f32>>) -> tensor<complex<f32>>
      "stablehlo.return"(%14) : (tensor<complex<f32>>) -> ()
    }) : (tensor<2x3xcomplex<f32>>, tensor<complex<f32>>) -> tensor<3xcomplex<f32>>
    "stablehlo.custom_call"(%5, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<3xcomplex<f32>>, tensor<3xcomplex<f32>>) -> ()
    "func.return"(%5) : (tensor<3xcomplex<f32>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x3xcomplex<f32>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[(1.45971477,-0.530904412), (3.91971016,1.38173568), (0.591264247,-0.429937601)], [(-4.28235674,6.83627414), (3.58333111,2.12905931), (0.620722293,3.66264701)]]> : tensor<2x3xcomplex<f32>>}> : () -> tensor<2x3xcomplex<f32>>
    "func.return"(%1) : (tensor<2x3xcomplex<f32>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3xcomplex<f32>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[(-4.28235674,6.83627414), (3.58333111,2.12905931), (0.591264247,-0.429937601)]> : tensor<3xcomplex<f32>>}> : () -> tensor<3xcomplex<f32>>
    "func.return"(%0) : (tensor<3xcomplex<f32>>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

