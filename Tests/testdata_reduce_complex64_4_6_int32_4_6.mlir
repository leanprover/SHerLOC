"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> (tensor<6xcomplex<f32>>, tensor<6xi32>), res_attrs = [{jax.result_info = "[0]", mhlo.layout_mode = "default"}, {jax.result_info = "[1]", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x6xcomplex<f32>>, tensor<4x6xi32>)
    %5:2 = "func.call"() <{callee = @expected}> : () -> (tensor<6xcomplex<f32>>, tensor<6xi32>)
    %6 = "stablehlo.constant"() <{value = dense<(3.000000e+00,0.000000e+00)> : tensor<complex<f32>>}> : () -> tensor<complex<f32>>
    %7 = "stablehlo.constant"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
    %8:2 = "stablehlo.reduce"(%4#0, %4#1, %6, %7) <{dimensions = array<i64: 0>}> ({
    ^bb0(%arg0: tensor<complex<f32>>, %arg1: tensor<i32>, %arg2: tensor<complex<f32>>, %arg3: tensor<i32>):
      %9 = "stablehlo.real"(%arg0) : (tensor<complex<f32>>) -> tensor<f32>
      %10 = "stablehlo.real"(%arg2) : (tensor<complex<f32>>) -> tensor<f32>
      %11 = "stablehlo.compare"(%9, %10) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %12 = "stablehlo.compare"(%9, %10) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GT>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %13 = "stablehlo.imag"(%arg0) : (tensor<complex<f32>>) -> tensor<f32>
      %14 = "stablehlo.imag"(%arg2) : (tensor<complex<f32>>) -> tensor<f32>
      %15 = "stablehlo.compare"(%13, %14) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GT>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %16 = "stablehlo.select"(%11, %15, %12) : (tensor<i1>, tensor<i1>, tensor<i1>) -> tensor<i1>
      %17 = "stablehlo.select"(%16, %arg0, %arg2) : (tensor<i1>, tensor<complex<f32>>, tensor<complex<f32>>) -> tensor<complex<f32>>
      %18 = "stablehlo.minimum"(%arg1, %arg3) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      "stablehlo.return"(%17, %18) : (tensor<complex<f32>>, tensor<i32>) -> ()
    }) : (tensor<4x6xcomplex<f32>>, tensor<4x6xi32>, tensor<complex<f32>>, tensor<i32>) -> (tensor<6xcomplex<f32>>, tensor<6xi32>)
    "stablehlo.custom_call"(%8#0, %5#0) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<6xcomplex<f32>>, tensor<6xcomplex<f32>>) -> ()
    "stablehlo.custom_call"(%8#1, %5#1) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<6xi32>, tensor<6xi32>) -> ()
    "func.return"(%8#0, %8#1) : (tensor<6xcomplex<f32>>, tensor<6xi32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x6xcomplex<f32>>, tensor<4x6xi32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %2 = "stablehlo.constant"() <{value = dense<[[(-0.381778359,-1.21680665), (-1.29195714,-5.36601257), (-3.14889767E-4,-1.04874849), (1.17392564,2.8273983), (0.539779663,1.59845591), (3.501680e+00,-1.58961439)], [(-8.656080e-02,-3.4427917), (1.70420969,1.87724805), (-2.88715482,1.74130571), (-1.96685147,2.44956136), (-0.395261019,0.492172718), (-2.60446668,-0.676473498)], [(-0.582556963,-7.35778141), (2.14501739,-2.86162686), (4.00450563,5.36438274), (0.776385069,-2.65484333), (-1.4155364,4.91823721), (-5.45412445,0.0295851938)], [(0.297839791,2.33191633), (4.9744482,-5.89148521), (-1.80037713,1.70301712), (-1.21536481,-5.40402555), (-0.411337882,1.20069647), (0.307315618,-2.6319325)]]> : tensor<4x6xcomplex<f32>>}> : () -> tensor<4x6xcomplex<f32>>
    %3 = "stablehlo.constant"() <{value = dense<[[4, -1, 0, 2, -1, 2], [0, -1, -4, 5, 0, 1], [-5, -2, -1, -2, 5, -4], [1, 0, -4, 0, 0, -1]]> : tensor<4x6xi32>}> : () -> tensor<4x6xi32>
    "func.return"(%2, %3) : (tensor<4x6xcomplex<f32>>, tensor<4x6xi32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<6xcomplex<f32>>, tensor<6xi32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[(3.000000e+00,0.000000e+00), (4.9744482,-5.89148521), (4.00450563,5.36438274), (3.000000e+00,0.000000e+00), (3.000000e+00,0.000000e+00), (3.501680e+00,-1.58961439)]> : tensor<6xcomplex<f32>>}> : () -> tensor<6xcomplex<f32>>
    %1 = "stablehlo.constant"() <{value = dense<[-5, -2, -4, -2, -1, -4]> : tensor<6xi32>}> : () -> tensor<6xi32>
    "func.return"(%0, %1) : (tensor<6xcomplex<f32>>, tensor<6xi32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

