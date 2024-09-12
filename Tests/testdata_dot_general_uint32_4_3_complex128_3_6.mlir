"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x6xcomplex<f64>>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x3xui32>, tensor<3x6xcomplex<f64>>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<4x6xcomplex<f64>>
    %5 = "stablehlo.convert"(%3#0) : (tensor<4x3xui32>) -> tensor<4x3xcomplex<f64>>
    %6 = "stablehlo.convert"(%3#1) : (tensor<3x6xcomplex<f64>>) -> tensor<3x6xcomplex<f64>>
    %7 = "stablehlo.dot_general"(%5, %6) <{dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>}> : (tensor<4x3xcomplex<f64>>, tensor<3x6xcomplex<f64>>) -> tensor<4x6xcomplex<f64>>
    "stablehlo.custom_call"(%7, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<4x6xcomplex<f64>>, tensor<4x6xcomplex<f64>>) -> ()
    "func.return"(%7) : (tensor<4x6xcomplex<f64>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x3xui32>, tensor<3x6xcomplex<f64>>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[1, 1, 3], [6, 0, 2], [1, 7, 3], [1, 2, 2]]> : tensor<4x3xui32>}> : () -> tensor<4x3xui32>
    %2 = "stablehlo.constant"() <{value = dense<[[(4.9236072523960308,0.57172272379854228), (0.68328283079253749,4.4203549750279674), (0.94212400840774913,3.7708706403528112), (-3.6181090270648903,-3.2441287400211367), (-3.7980835368316028,-0.54696812409477269), (1.1412120017864351,2.4492524311206947)], [(2.9208743974920544,3.9642423502344903), (7.4995904267297551,0.072242571714250053), (-1.7920234203056595,0.79972836783487189), (-0.62215742926722939,2.67621650060648), (-1.7610228099893588,1.6408554521141911), (-0.89131392306348855,-0.13975238201150125)], [(5.1677162090791571,1.0479317760988078), (-4.1925263192968689,-1.8501032724736928), (0.035028655371740089,1.5618010160326734), (-0.11898110763828931,-1.2091172710708229), (1.0931198624881222,-0.34968837412081821), (0.65536470509319478,1.2896935216495642)]]> : tensor<3x6xcomplex<f64>>}> : () -> tensor<3x6xcomplex<f64>>
    "func.return"(%1, %2) : (tensor<4x3xui32>, tensor<3x6xcomplex<f64>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xcomplex<f64>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[(23.347630277125557,7.6797604023294567), (-4.3947057003683145,-1.0577122706788611), (-0.74481344578269015,9.2560020562857037), (-4.5972097792469881,-4.1952640526271257), (-2.2797467593565948,0.044822205656963821), (2.2159921940025309,6.1785806140578856)], [(39.877075932534495,5.5261998949888689), (-4.2853556538385131,22.82192330522042), (5.7228013611899753,25.748825874182213), (-21.946616377665922,-21.883006982268469), (-20.602261496013373,-3.9811854928102726), (8.1580014209050002,17.274901630023297)], [(40.872876662077886,31.465214503736401), (40.602836860010214,-0.62425684039336105), (-11.496953967616648,14.054372263294935), (-8.3301543548503645,11.862034951011756), (-12.845883619292746,9.8899549183421094), (-3.1318913443784,5.3400663219888784)], [(21.100788465538454,10.596070976465137), (7.2974110456583094,0.86463357350908154), (-2.5718655214600901,8.4939294080879008), (-5.1003861008759275,-0.30993028094982256), (-5.133889431834076,2.0353660318919733), (0.66931356584584756,4.7491347103968202)]]> : tensor<4x6xcomplex<f64>>}> : () -> tensor<4x6xcomplex<f64>>
    "func.return"(%0) : (tensor<4x6xcomplex<f64>>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

