"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> (tensor<100xi32>, tensor<100xi32>, tensor<100xf32>), res_attrs = [{jax.result_info = "[0]", mhlo.layout_mode = "default"}, {jax.result_info = "[1]", mhlo.layout_mode = "default"}, {jax.result_info = "[2]", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %6:3 = "func.call"() <{callee = @inputs}> : () -> (tensor<100xi32>, tensor<100xi32>, tensor<100xf32>)
    %7:3 = "func.call"() <{callee = @expected}> : () -> (tensor<100xi32>, tensor<100xi32>, tensor<100xf32>)
    %8:3 = "stablehlo.sort"(%6#0, %6#1, %6#2) <{dimension = 0 : i64, is_stable = true}> ({
    ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<f32>, %arg5: tensor<f32>):
      %9 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %10 = "stablehlo.compare"(%arg4, %9) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %11 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %12 = "stablehlo.select"(%10, %11, %arg4) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
      %13 = "stablehlo.compare"(%arg4, %arg4) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %14 = "stablehlo.constant"() <{value = dense<0x7FC00000> : tensor<f32>}> : () -> tensor<f32>
      %15 = "stablehlo.select"(%13, %14, %12) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
      %16 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %17 = "stablehlo.compare"(%arg5, %16) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %18 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %19 = "stablehlo.select"(%17, %18, %arg5) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
      %20 = "stablehlo.compare"(%arg5, %arg5) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %21 = "stablehlo.constant"() <{value = dense<0x7FC00000> : tensor<f32>}> : () -> tensor<f32>
      %22 = "stablehlo.select"(%20, %21, %19) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
      %23 = "stablehlo.compare"(%15, %22) <{compare_type = #stablehlo<comparison_type TOTALORDER>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %24 = "stablehlo.compare"(%arg2, %arg3) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %25 = "stablehlo.compare"(%arg2, %arg3) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %26 = "stablehlo.and"(%25, %23) : (tensor<i1>, tensor<i1>) -> tensor<i1>
      %27 = "stablehlo.or"(%24, %26) : (tensor<i1>, tensor<i1>) -> tensor<i1>
      %28 = "stablehlo.compare"(%arg0, %arg1) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %29 = "stablehlo.compare"(%arg0, %arg1) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %30 = "stablehlo.and"(%29, %27) : (tensor<i1>, tensor<i1>) -> tensor<i1>
      %31 = "stablehlo.or"(%28, %30) : (tensor<i1>, tensor<i1>) -> tensor<i1>
      "stablehlo.return"(%31) : (tensor<i1>) -> ()
    }) : (tensor<100xi32>, tensor<100xi32>, tensor<100xf32>) -> (tensor<100xi32>, tensor<100xi32>, tensor<100xf32>)
    "stablehlo.custom_call"(%8#0, %7#0) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<100xi32>, tensor<100xi32>) -> ()
    "stablehlo.custom_call"(%8#1, %7#1) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<100xi32>, tensor<100xi32>) -> ()
    "stablehlo.custom_call"(%8#2, %7#2) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<100xf32>, tensor<100xf32>) -> ()
    "func.return"(%8#0, %8#1, %8#2) : (tensor<100xi32>, tensor<100xi32>, tensor<100xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<100xi32>, tensor<100xi32>, tensor<100xf32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1]> : tensor<100xi32>}> : () -> tensor<100xi32>
    %4 = "stablehlo.constant"() <{value = dense<[1, 5, 2, 3, 4, 2, -5, 0, 2, 2, 0, -1, -1, 0, 0, -5, 4, -3, 3, 0, 0, 1, 0, 5, 3, -3, 0, 1, 0, 0, 0, -2, 0, 4, -2, -1, -3, 0, -1, 3, 1, -4, -4, -2, 3, 5, 1, 7, 2, 2, 1, 0, -3, 1, -1, -1, -3, -3, -1, -2, 2, -5, 0, 0, -1, -1, -1, -2, 1, -2, 2, 3, 0, -2, 4, -3, 0, 0, -4, 0, 1, 0, -2, 3, 0, -2, -1, -5, 1, 6, -2, 1, -3, 1, 0, 0, 0, 3, -1, 0]> : tensor<100xi32>}> : () -> tensor<100xi32>
    %5 = "stablehlo.constant"() <{value = dense<[6.20356131, 5.65439892, 2.38549161, -3.96550655, -3.96124077, 1.99819422, -4.68467474, -2.49656415, -2.54703045, 0.0197906531, -0.740306317, -3.19704342, 1.27773118, -1.63176405, -1.01926863, 1.9354341, -2.4787395, 2.25621748, -1.48426414, -0.342727482, -0.651743412, 1.14231288, 2.13769197, 5.721990e-01, 3.78986549, 4.27968693, 0.455839902, -1.66110146, -2.93755436, -5.64413548, 2.54588413, 8.145430e-01, 0.273220271, 0.788339495, 2.00295901, -1.60751247, -1.70372462, -1.1993798, 3.84779954, -0.852602064, 3.09254956, 1.50013828, 0.799096703, -3.54357862, 0.281735957, 1.62879372, 1.44648623, 0.415672541, -4.84510756, 3.98790479, -3.5780375, -6.54678869, 2.99260116, -1.87401211, -0.621982455, 1.01716506, 2.95774937, 3.05121779, -0.028854521, -2.60105753, 1.91555417, -7.83330297, -4.62947416, -0.03703003, 3.06019616, -0.196689233, -2.59691501, -2.38492537, 4.45763779, -0.077415429, -3.38808942, -1.14041936, -0.225846633, -3.38728166, 4.22935104, 2.17284632, -3.660960e+00, -1.2449944, 2.02476263, -2.42962742, 2.82782459, 1.50918257, -2.22166085, -2.52914572, 1.04657984, -1.03708935, -1.80974913, 1.55136991, -0.658415257, 1.07587576, 5.95514774, -0.463521421, 5.02912045, -1.08982968, -0.883667945, 1.4935534, -0.537910938, -3.090550e-01, 2.59730196, -1.88129139]> : tensor<100xf32>}> : () -> tensor<100xf32>
    "func.return"(%3, %4, %5) : (tensor<100xi32>, tensor<100xi32>, tensor<100xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<100xi32>, tensor<100xi32>, tensor<100xf32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]> : tensor<100xi32>}> : () -> tensor<100xi32>
    %1 = "stablehlo.constant"() <{value = dense<[-5, -4, -3, -3, -3, -3, -2, -2, -2, -2, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 5, 5, 5, 7, -5, -5, -5, -4, -4, -3, -3, -3, -3, -2, -2, -2, -2, -2, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 6]> : tensor<100xi32>}> : () -> tensor<100xi32>
    %2 = "stablehlo.constant"() <{value = dense<[1.55136991, 0.799096703, 2.17284632, 2.25621748, 3.05121779, 5.02912045, -2.60105753, -2.38492537, -1.03708935, 5.95514774, -0.028854521, 1.01716506, 1.27773118, -4.62947416, -2.49656415, -2.42962742, -1.63176405, -0.883667945, -0.740306317, -0.651743412, -0.342727482, -0.225846633, 0.273220271, 0.455839902, 1.4935534, 2.54588413, -3.5780375, -1.66110146, -1.08982968, -0.658415257, 1.14231288, 2.82782459, 4.45763779, -4.84510756, 1.91555417, 2.38549161, -3.96550655, -2.52914572, -1.14041936, 0.281735957, 4.22935104, 5.721990e-01, 1.62879372, 5.65439892, 0.415672541, -7.83330297, -4.68467474, 1.9354341, 1.50013828, 2.02476263, -1.70372462, 2.95774937, 2.99260116, 4.27968693, -3.54357862, -3.38728166, -2.22166085, -0.077415429, 8.145430e-01, 2.00295901, -3.19704342, -2.59691501, -1.80974913, -1.60751247, -0.621982455, -0.196689233, 2.59730196, 3.06019616, 3.84779954, -6.54678869, -5.64413548, -3.660960e+00, -2.93755436, -1.88129139, -1.2449944, -1.1993798, -1.01926863, -0.537910938, -0.03703003, 1.04657984, 1.50918257, 2.13769197, -1.87401211, -0.463521421, 1.44648623, 3.09254956, 6.20356131, -3.38808942, -2.54703045, 0.0197906531, 1.99819422, 3.98790479, -1.48426414, -0.852602064, -3.090550e-01, 3.78986549, -3.96124077, -2.4787395, 0.788339495, 1.07587576]> : tensor<100xf32>}> : () -> tensor<100xf32>
    "func.return"(%0, %1, %2) : (tensor<100xi32>, tensor<100xi32>, tensor<100xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

