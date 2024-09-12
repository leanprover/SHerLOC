"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x3xcomplex<f64>>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<2x3xcomplex<f64>>, tensor<2x3xcomplex<f64>>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<4x3xcomplex<f64>>
    %5 = "stablehlo.concatenate"(%3#0, %3#1) <{dimension = 0 : i64}> : (tensor<2x3xcomplex<f64>>, tensor<2x3xcomplex<f64>>) -> tensor<4x3xcomplex<f64>>
    "stablehlo.custom_call"(%5, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<4x3xcomplex<f64>>, tensor<4x3xcomplex<f64>>) -> ()
    "func.return"(%5) : (tensor<4x3xcomplex<f64>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<2x3xcomplex<f64>>, tensor<2x3xcomplex<f64>>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[(0.12093412311353263,0.38010149284511113), (-1.9093750408113137,1.1698267132297944), (3.3517893977389743,-5.5234094816836858)], [(-1.058133517624317,-4.1777904231767859), (1.6058419047578061,-1.8533674523699413), (-4.0672779033135003,6.3303697566098354)]]> : tensor<2x3xcomplex<f64>>}> : () -> tensor<2x3xcomplex<f64>>
    %2 = "stablehlo.constant"() <{value = dense<[[(-0.23924756010979267,1.6093707259284415), (6.5674842892777754,-2.0374352217249356), (-3.8670516047106598,-2.6247801011694483)], [(2.9389038494109894,-3.1402112199128149), (3.7405834733626224,-5.3309082802223058), (0.54943704590034437,2.2843780882730322)]]> : tensor<2x3xcomplex<f64>>}> : () -> tensor<2x3xcomplex<f64>>
    "func.return"(%1, %2) : (tensor<2x3xcomplex<f64>>, tensor<2x3xcomplex<f64>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x3xcomplex<f64>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[(0.12093412311353263,0.38010149284511113), (-1.9093750408113137,1.1698267132297944), (3.3517893977389743,-5.5234094816836858)], [(-1.058133517624317,-4.1777904231767859), (1.6058419047578061,-1.8533674523699413), (-4.0672779033135003,6.3303697566098354)], [(-0.23924756010979267,1.6093707259284415), (6.5674842892777754,-2.0374352217249356), (-3.8670516047106598,-2.6247801011694483)], [(2.9389038494109894,-3.1402112199128149), (3.7405834733626224,-5.3309082802223058), (0.54943704590034437,2.2843780882730322)]]> : tensor<4x3xcomplex<f64>>}> : () -> tensor<4x3xcomplex<f64>>
    "func.return"(%0) : (tensor<4x3xcomplex<f64>>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

