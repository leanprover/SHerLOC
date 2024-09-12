"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3x5xbf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x6xbf16>, tensor<4x6xbf16>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<3x5xbf16>
    %5 = "stablehlo.constant"() <{value = dense<0x7F80> : tensor<bf16>}> : () -> tensor<bf16>
    %6 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<bf16>}> : () -> tensor<bf16>
    %7:2 = "stablehlo.reduce_window"(%3#1, %3#0, %5, %6) <{window_dimensions = array<i64: 2, 2>}> ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>, %arg2: tensor<bf16>, %arg3: tensor<bf16>):
      %8 = "stablehlo.compare"(%arg0, %arg2) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LE>}> : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
      %9 = "stablehlo.select"(%8, %arg0, %arg2) : (tensor<i1>, tensor<bf16>, tensor<bf16>) -> tensor<bf16>
      %10 = "stablehlo.select"(%8, %arg1, %arg3) : (tensor<i1>, tensor<bf16>, tensor<bf16>) -> tensor<bf16>
      "stablehlo.return"(%9, %10) : (tensor<bf16>, tensor<bf16>) -> ()
    }) : (tensor<4x6xbf16>, tensor<4x6xbf16>, tensor<bf16>, tensor<bf16>) -> (tensor<3x5xbf16>, tensor<3x5xbf16>)
    "stablehlo.custom_call"(%7#1, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<3x5xbf16>, tensor<3x5xbf16>) -> ()
    "func.return"(%7#1) : (tensor<3x5xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x6xbf16>, tensor<4x6xbf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[-1.296880e+00, 2.671880e+00, 2.046880e+00, -3.375000e+00, -1.515630e+00, 2.031250e+00], [-3.171880e+00, -3.656250e+00, 2.640630e+00, -3.156250e+00, -4.062500e+00, -1.166990e-01], [8.750000e+00, -2.953130e+00, 3.453130e+00, 1.492190e+00, -5.062500e+00, 1.617190e+00], [1.671880e+00, -8.164060e-01, 1.078130e+00, 1.632810e+00, 3.187500e+00, -1.156250e+00]]> : tensor<4x6xbf16>}> : () -> tensor<4x6xbf16>
    %2 = "stablehlo.constant"() <{value = dense<[[-3.265630e+00, -2.375000e+00, -1.804690e+00, -7.062500e+00, -2.640630e+00, -6.591800e-02], [4.062500e-01, 6.210940e-01, 4.062500e+00, 1.765630e+00, -3.015630e+00, -1.375000e+00], [3.484380e+00, -5.406250e+00, -2.250000e+00, 1.039060e+00, -9.687500e-01, -2.375000e+00], [-4.250000e+00, 8.554680e-01, -1.343750e+00, -6.031250e+00, -2.359380e+00, -1.187500e+00]]> : tensor<4x6xbf16>}> : () -> tensor<4x6xbf16>
    "func.return"(%1, %2) : (tensor<4x6xbf16>, tensor<4x6xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x5xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[-1.296880e+00, 2.671880e+00, -3.375000e+00, -3.375000e+00, -4.062500e+00], [-2.953130e+00, -2.953130e+00, 3.453130e+00, -4.062500e+00, -4.062500e+00], [-2.953130e+00, -2.953130e+00, 1.632810e+00, 1.632810e+00, 1.617190e+00]]> : tensor<3x5xbf16>}> : () -> tensor<3x5xbf16>
    "func.return"(%0) : (tensor<3x5xbf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

