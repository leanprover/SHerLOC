"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<5x7xf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<5x7xf16>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<5x7xf16>
    %4 = "stablehlo.reduce_precision"(%2) <{exponent_bits = 11 : i32, mantissa_bits = 52 : i32}> : (tensor<5x7xf16>) -> tensor<5x7xf16>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<5x7xf16>, tensor<5x7xf16>) -> ()
    "func.return"(%4) : (tensor<5x7xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x7xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[-1.533200e+00, -5.031250e+00, -4.355470e+00, -1.416020e+00, -1.691410e+00, -9.980460e-01, -1.057620e+00], [6.386710e+00, 9.755850e-01, 1.442380e+00, 3.042970e+00, -3.447270e+00, -2.392580e-01, 3.384770e+00], [1.333010e-01, -2.309570e-01, 1.952150e+00, 4.511720e+00, 2.937500e+00, 2.779300e+00, -7.647710e-02], [4.785160e+00, -2.441410e+00, 2.738280e+00, 4.046880e+00, 3.496090e+00, 4.386720e+00, -3.058590e+00], [9.848630e-01, -1.439450e+00, 1.610350e+00, 5.605460e-01, -6.552730e-01, -4.820310e+00, 3.820800e-02]]> : tensor<5x7xf16>}> : () -> tensor<5x7xf16>
    "func.return"(%1) : (tensor<5x7xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x7xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[-1.533200e+00, -5.031250e+00, -4.355470e+00, -1.416020e+00, -1.691410e+00, -9.980460e-01, -1.057620e+00], [6.386710e+00, 9.755850e-01, 1.442380e+00, 3.042970e+00, -3.447270e+00, -2.392580e-01, 3.384770e+00], [1.333010e-01, -2.309570e-01, 1.952150e+00, 4.511720e+00, 2.937500e+00, 2.779300e+00, -7.647710e-02], [4.785160e+00, -2.441410e+00, 2.738280e+00, 4.046880e+00, 3.496090e+00, 4.386720e+00, -3.058590e+00], [9.848630e-01, -1.439450e+00, 1.610350e+00, 5.605460e-01, -6.552730e-01, -4.820310e+00, 3.820800e-02]]> : tensor<5x7xf16>}> : () -> tensor<5x7xf16>
    "func.return"(%0) : (tensor<5x7xf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

