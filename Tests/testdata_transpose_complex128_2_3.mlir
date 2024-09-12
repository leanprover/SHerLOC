"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3x2xcomplex<f64>>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<2x3xcomplex<f64>>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<3x2xcomplex<f64>>
    %4 = "stablehlo.transpose"(%2) <{permutation = array<i64: 1, 0>}> : (tensor<2x3xcomplex<f64>>) -> tensor<3x2xcomplex<f64>>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<3x2xcomplex<f64>>, tensor<3x2xcomplex<f64>>) -> ()
    "func.return"(%4) : (tensor<3x2xcomplex<f64>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x3xcomplex<f64>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[(-1.3119832852479361,-2.7942567775397702), (-5.2463848304345424,-0.96527108533955164), (0.81592895582045633,-3.094400866203606)], [(1.2780774567730127,-1.2231556133313144), (-2.2705888602008182,-0.24873578516395517), (-4.0748643883366746,-0.070090492007976304)]]> : tensor<2x3xcomplex<f64>>}> : () -> tensor<2x3xcomplex<f64>>
    "func.return"(%1) : (tensor<2x3xcomplex<f64>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x2xcomplex<f64>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[(-1.3119832852479361,-2.7942567775397702), (1.2780774567730127,-1.2231556133313144)], [(-5.2463848304345424,-0.96527108533955164), (-2.2705888602008182,-0.24873578516395517)], [(0.81592895582045633,-3.094400866203606), (-4.0748643883366746,-0.070090492007976304)]]> : tensor<3x2xcomplex<f64>>}> : () -> tensor<3x2xcomplex<f64>>
    "func.return"(%0) : (tensor<3x2xcomplex<f64>>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

