"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<18xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %4:3 = "func.call"() <{callee = @inputs}> : () -> (tensor<i1>, tensor<18xf32>, tensor<18xf32>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<18xf32>
    %6 = "stablehlo.select"(%4#0, %4#2, %4#1) : (tensor<i1>, tensor<18xf32>, tensor<18xf32>) -> tensor<18xf32>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<18xf32>, tensor<18xf32>) -> ()
    "func.return"(%6) : (tensor<18xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<i1>, tensor<18xf32>, tensor<18xf32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[-1.83992159, -5.53955698, -0.79379636, 1.26131332, 1.52134728, -2.55310225, -5.83623314, -2.70676374, -1.89610958, 5.02900076, 1.20761502, -1.07989526, -0.395469189, -2.707490e+00, 1.10248196, -0.0470473804, -1.68658864, -1.51551247]> : tensor<18xf32>}> : () -> tensor<18xf32>
    %2 = "stablehlo.constant"() <{value = dense<[4.03491116, 4.54813766, 0.525275111, 6.68681145, -4.24286127, 7.29083967, -1.08588982, 1.0339278, 0.348950595, -3.03581524, -2.31413841, 1.69352174, -1.42940807, 5.65003777, 6.60452461, 2.04546762, 3.83973241, 0.686347365]> : tensor<18xf32>}> : () -> tensor<18xf32>
    %3 = "stablehlo.constant"() <{value = dense<true> : tensor<i1>}> : () -> tensor<i1>
    "func.return"(%3, %1, %2) : (tensor<i1>, tensor<18xf32>, tensor<18xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<18xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[4.03491116, 4.54813766, 0.525275111, 6.68681145, -4.24286127, 7.29083967, -1.08588982, 1.0339278, 0.348950595, -3.03581524, -2.31413841, 1.69352174, -1.42940807, 5.65003777, 6.60452461, 2.04546762, 3.83973241, 0.686347365]> : tensor<18xf32>}> : () -> tensor<18xf32>
    "func.return"(%0) : (tensor<18xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

