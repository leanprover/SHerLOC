"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x6xcomplex<f64>>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x3xui16>, tensor<3x6xcomplex<f64>>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<4x6xcomplex<f64>>
    %5 = "stablehlo.convert"(%3#0) : (tensor<4x3xui16>) -> tensor<4x3xcomplex<f64>>
    %6 = "stablehlo.convert"(%3#1) : (tensor<3x6xcomplex<f64>>) -> tensor<3x6xcomplex<f64>>
    %7 = "stablehlo.dot_general"(%5, %6) <{dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>}> : (tensor<4x3xcomplex<f64>>, tensor<3x6xcomplex<f64>>) -> tensor<4x6xcomplex<f64>>
    "stablehlo.custom_call"(%7, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<4x6xcomplex<f64>>, tensor<4x6xcomplex<f64>>) -> ()
    "func.return"(%7) : (tensor<4x6xcomplex<f64>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x3xui16>, tensor<3x6xcomplex<f64>>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[1, 3, 1], [6, 0, 1], [0, 6, 1], [0, 2, 2]]> : tensor<4x3xui16>}> : () -> tensor<4x3xui16>
    %2 = "stablehlo.constant"() <{value = dense<[[(1.280173890647627,-0.8613100076781276), (0.0097981891182064886,1.5766054523761788), (2.5714961511757348,0.075671569118278215), (-1.7241556098865549,0.49937133174992854), (-1.2378194583323094,-1.6808325868492264), (2.085160797921648,-2.1422901939938166)], [(1.2997485829559419,-4.4123648564208384), (4.2304738921033174,-1.3994199955975479), (-0.33740363593417544,-0.036170333631881178), (-2.8068314477856706,0.8311321473113924), (-10.600406999191266,5.1727077847233147), (-2.3346347489189867,1.5518577661671276)], [(-0.80348501966780849,-4.6134657178325833), (2.6867497788674481,-2.9898124399478441), (0.49098036658241095,-3.1523078073136981), (1.6713740530579035,1.2781190815188921), (-1.7040955590320068,1.8804005591033057), (2.0251927181709322,1.8567724896788462)]]> : tensor<3x6xcomplex<f64>>}> : () -> tensor<3x6xcomplex<f64>>
    "func.return"(%1, %2) : (tensor<4x3xui16>, tensor<3x6xcomplex<f64>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xcomplex<f64>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[(4.3759346198476443,-18.711870294773227), (15.387969644295607,-5.6114669743643093), (2.0502656099556194,-3.1851472390910636), (-8.4732759001856639,4.2708868552029973), (-34.743136014938116,15.717691326424024), (-2.89355073066438,4.3700555941864119)], [(6.8775583242179525,-9.78132576390135), (2.745538913576687,6.4698202743092281), (15.91995727363682,-2.6982783926040286), (-8.673559606261426,4.2743470720184638), (-9.1310123090258628,-8.2045949619920524), (14.536157505700821,-10.996968674284053)], [(6.9950064780678431,-31.087654856357613), (28.069593131487352,-11.386332413533133), (-1.5334414490226416,-3.3693298091049853), (-15.169614633656121,6.2649119653872463), (-65.306537554179613,32.916647267443196), (-11.982615775342989,11.167919086681611)], [(0.99252712657626673,-18.051661148506845), (13.83444734194153,-8.7784648710907831), (0.30715346129647103,-6.3769562818911583), (-2.2709147894555342,4.218502457660569), (-24.609005116446546,14.106216687653241), (-0.61888406149610908,6.8172605116919476)]]> : tensor<4x6xcomplex<f64>>}> : () -> tensor<4x6xcomplex<f64>>
    "func.return"(%0) : (tensor<4x6xcomplex<f64>>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

