"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3x5xf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x6xf16>, tensor<4x6xf16>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<3x5xf16>
    %5 = "stablehlo.constant"() <{value = dense<0x7C00> : tensor<f16>}> : () -> tensor<f16>
    %6 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f16>}> : () -> tensor<f16>
    %7:2 = "stablehlo.reduce_window"(%3#1, %3#0, %5, %6) <{window_dimensions = array<i64: 2, 2>}> ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>, %arg2: tensor<f16>, %arg3: tensor<f16>):
      %8 = "stablehlo.compare"(%arg0, %arg2) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LE>}> : (tensor<f16>, tensor<f16>) -> tensor<i1>
      %9 = "stablehlo.select"(%8, %arg0, %arg2) : (tensor<i1>, tensor<f16>, tensor<f16>) -> tensor<f16>
      %10 = "stablehlo.select"(%8, %arg1, %arg3) : (tensor<i1>, tensor<f16>, tensor<f16>) -> tensor<f16>
      "stablehlo.return"(%9, %10) : (tensor<f16>, tensor<f16>) -> ()
    }) : (tensor<4x6xf16>, tensor<4x6xf16>, tensor<f16>, tensor<f16>) -> (tensor<3x5xf16>, tensor<3x5xf16>)
    "stablehlo.custom_call"(%7#1, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<3x5xf16>, tensor<3x5xf16>) -> ()
    "func.return"(%7#1) : (tensor<3x5xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x6xf16>, tensor<4x6xf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[2.392580e+00, 3.939450e+00, -5.648440e+00, 8.652340e-01, 1.454100e+00, 6.093750e-01], [3.189450e+00, -7.421880e-01, -1.119140e+00, -2.177730e+00, 2.199220e+00, -7.075190e-01], [4.699710e-01, -8.984370e-01, -1.866210e+00, 7.580560e-02, -7.613280e+00, -7.242180e+00], [4.109380e+00, 3.443360e+00, 2.589840e+00, 1.947270e+00, -3.361820e-01, -1.671880e+00]]> : tensor<4x6xf16>}> : () -> tensor<4x6xf16>
    %2 = "stablehlo.constant"() <{value = dense<[[2.197270e+00, 1.117190e+00, 3.562010e-01, 3.738280e+00, -2.197270e+00, 1.822270e+00], [8.984370e-01, -8.282470e-02, -1.631840e+00, -5.304690e+00, -2.890630e+00, 3.591800e+00], [5.402340e+00, 1.822270e+00, 7.626950e-01, 4.664060e+00, -9.718750e+00, -1.122070e+00], [3.025390e+00, 1.737060e-01, 1.925050e-01, -7.617180e-01, 4.625000e+00, 1.282230e+00]]> : tensor<4x6xf16>}> : () -> tensor<4x6xf16>
    "func.return"(%1, %2) : (tensor<4x6xf16>, tensor<4x6xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x5xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[-7.421880e-01, -1.119140e+00, -2.177730e+00, -2.177730e+00, 2.199220e+00], [-7.421880e-01, -1.119140e+00, -2.177730e+00, -7.613280e+00, -7.613280e+00], [3.443360e+00, 3.443360e+00, 1.947270e+00, -7.613280e+00, -7.613280e+00]]> : tensor<3x5xf16>}> : () -> tensor<3x5xf16>
    "func.return"(%0) : (tensor<3x5xf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

