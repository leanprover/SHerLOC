"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<20x20xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<20x20xf32>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<20x20xf32>
    %4 = "stablehlo.abs"(%2) : (tensor<20x20xf32>) -> tensor<20x20xf32>
    %5 = "stablehlo.constant"() <{value = dense<5.000000e-01> : tensor<f32>}> : () -> tensor<f32>
    %6 = "stablehlo.broadcast_in_dim"(%5) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %7 = "stablehlo.constant"() <{value = dense<2.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %8 = "stablehlo.broadcast_in_dim"(%7) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %9 = "stablehlo.constant"() <{value = dense<3.200000e+01> : tensor<f32>}> : () -> tensor<f32>
    %10 = "stablehlo.broadcast_in_dim"(%9) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %11 = "stablehlo.multiply"(%6, %4) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %12 = "stablehlo.subtract"(%11, %8) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %13 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %14 = "stablehlo.broadcast_in_dim"(%13) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %15 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %16 = "stablehlo.broadcast_in_dim"(%15) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %17 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %18 = "stablehlo.broadcast_in_dim"(%17) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %19 = "stablehlo.multiply"(%12, %14) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %20 = "stablehlo.subtract"(%19, %16) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %21 = "stablehlo.constant"() <{value = dense<-1.300025009986248E-8> : tensor<f64>}> : () -> tensor<f64>
    %22 = "stablehlo.convert"(%21) : (tensor<f64>) -> tensor<f32>
    %23 = "stablehlo.broadcast_in_dim"(%22) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %24 = "stablehlo.add"(%20, %23) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %25 = "stablehlo.multiply"(%12, %24) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %26 = "stablehlo.subtract"(%25, %14) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %27 = "stablehlo.constant"() <{value = dense<6.0469950225419186E-8> : tensor<f64>}> : () -> tensor<f64>
    %28 = "stablehlo.convert"(%27) : (tensor<f64>) -> tensor<f32>
    %29 = "stablehlo.broadcast_in_dim"(%28) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %30 = "stablehlo.add"(%26, %29) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %31 = "stablehlo.multiply"(%12, %30) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %32 = "stablehlo.subtract"(%31, %24) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %33 = "stablehlo.constant"() <{value = dense<-2.6707938539406119E-7> : tensor<f64>}> : () -> tensor<f64>
    %34 = "stablehlo.convert"(%33) : (tensor<f64>) -> tensor<f32>
    %35 = "stablehlo.broadcast_in_dim"(%34) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %36 = "stablehlo.add"(%32, %35) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %37 = "stablehlo.multiply"(%12, %36) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %38 = "stablehlo.subtract"(%37, %30) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %39 = "stablehlo.constant"() <{value = dense<1.1173875391201037E-6> : tensor<f64>}> : () -> tensor<f64>
    %40 = "stablehlo.convert"(%39) : (tensor<f64>) -> tensor<f32>
    %41 = "stablehlo.broadcast_in_dim"(%40) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %42 = "stablehlo.add"(%38, %41) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %43 = "stablehlo.multiply"(%12, %42) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %44 = "stablehlo.subtract"(%43, %36) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %45 = "stablehlo.constant"() <{value = dense<-4.4167383584587505E-6> : tensor<f64>}> : () -> tensor<f64>
    %46 = "stablehlo.convert"(%45) : (tensor<f64>) -> tensor<f32>
    %47 = "stablehlo.broadcast_in_dim"(%46) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %48 = "stablehlo.add"(%44, %47) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %49 = "stablehlo.multiply"(%12, %48) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %50 = "stablehlo.subtract"(%49, %42) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %51 = "stablehlo.constant"() <{value = dense<1.6448448070728896E-5> : tensor<f64>}> : () -> tensor<f64>
    %52 = "stablehlo.convert"(%51) : (tensor<f64>) -> tensor<f32>
    %53 = "stablehlo.broadcast_in_dim"(%52) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %54 = "stablehlo.add"(%50, %53) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %55 = "stablehlo.multiply"(%12, %54) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %56 = "stablehlo.subtract"(%55, %48) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %57 = "stablehlo.constant"() <{value = dense<-5.754195010082104E-5> : tensor<f64>}> : () -> tensor<f64>
    %58 = "stablehlo.convert"(%57) : (tensor<f64>) -> tensor<f32>
    %59 = "stablehlo.broadcast_in_dim"(%58) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %60 = "stablehlo.add"(%56, %59) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %61 = "stablehlo.multiply"(%12, %60) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %62 = "stablehlo.subtract"(%61, %54) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %63 = "stablehlo.constant"() <{value = dense<1.8850288509584165E-4> : tensor<f64>}> : () -> tensor<f64>
    %64 = "stablehlo.convert"(%63) : (tensor<f64>) -> tensor<f32>
    %65 = "stablehlo.broadcast_in_dim"(%64) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %66 = "stablehlo.add"(%62, %65) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %67 = "stablehlo.multiply"(%12, %66) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %68 = "stablehlo.subtract"(%67, %60) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %69 = "stablehlo.constant"() <{value = dense<-5.7637557453858236E-4> : tensor<f64>}> : () -> tensor<f64>
    %70 = "stablehlo.convert"(%69) : (tensor<f64>) -> tensor<f32>
    %71 = "stablehlo.broadcast_in_dim"(%70) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %72 = "stablehlo.add"(%68, %71) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %73 = "stablehlo.multiply"(%12, %72) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %74 = "stablehlo.subtract"(%73, %66) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %75 = "stablehlo.constant"() <{value = dense<0.0016394756169413357> : tensor<f64>}> : () -> tensor<f64>
    %76 = "stablehlo.convert"(%75) : (tensor<f64>) -> tensor<f32>
    %77 = "stablehlo.broadcast_in_dim"(%76) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %78 = "stablehlo.add"(%74, %77) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %79 = "stablehlo.multiply"(%12, %78) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %80 = "stablehlo.subtract"(%79, %72) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %81 = "stablehlo.constant"() <{value = dense<-0.0043243099950505759> : tensor<f64>}> : () -> tensor<f64>
    %82 = "stablehlo.convert"(%81) : (tensor<f64>) -> tensor<f32>
    %83 = "stablehlo.broadcast_in_dim"(%82) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %84 = "stablehlo.add"(%80, %83) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %85 = "stablehlo.multiply"(%12, %84) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %86 = "stablehlo.subtract"(%85, %78) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %87 = "stablehlo.constant"() <{value = dense<0.010546460394594998> : tensor<f64>}> : () -> tensor<f64>
    %88 = "stablehlo.convert"(%87) : (tensor<f64>) -> tensor<f32>
    %89 = "stablehlo.broadcast_in_dim"(%88) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %90 = "stablehlo.add"(%86, %89) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %91 = "stablehlo.multiply"(%12, %90) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %92 = "stablehlo.subtract"(%91, %84) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %93 = "stablehlo.constant"() <{value = dense<-0.023737414805899471> : tensor<f64>}> : () -> tensor<f64>
    %94 = "stablehlo.convert"(%93) : (tensor<f64>) -> tensor<f32>
    %95 = "stablehlo.broadcast_in_dim"(%94) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %96 = "stablehlo.add"(%92, %95) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %97 = "stablehlo.multiply"(%12, %96) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %98 = "stablehlo.subtract"(%97, %90) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %99 = "stablehlo.constant"() <{value = dense<0.049305284239670712> : tensor<f64>}> : () -> tensor<f64>
    %100 = "stablehlo.convert"(%99) : (tensor<f64>) -> tensor<f32>
    %101 = "stablehlo.broadcast_in_dim"(%100) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %102 = "stablehlo.add"(%98, %101) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %103 = "stablehlo.multiply"(%12, %102) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %104 = "stablehlo.subtract"(%103, %96) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %105 = "stablehlo.constant"() <{value = dense<-0.094901097048047639> : tensor<f64>}> : () -> tensor<f64>
    %106 = "stablehlo.convert"(%105) : (tensor<f64>) -> tensor<f32>
    %107 = "stablehlo.broadcast_in_dim"(%106) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %108 = "stablehlo.add"(%104, %107) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %109 = "stablehlo.multiply"(%12, %108) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %110 = "stablehlo.subtract"(%109, %102) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %111 = "stablehlo.constant"() <{value = dense<0.17162090152220877> : tensor<f64>}> : () -> tensor<f64>
    %112 = "stablehlo.convert"(%111) : (tensor<f64>) -> tensor<f32>
    %113 = "stablehlo.broadcast_in_dim"(%112) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %114 = "stablehlo.add"(%110, %113) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %115 = "stablehlo.multiply"(%12, %114) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %116 = "stablehlo.subtract"(%115, %108) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %117 = "stablehlo.constant"() <{value = dense<-0.3046826723431984> : tensor<f64>}> : () -> tensor<f64>
    %118 = "stablehlo.convert"(%117) : (tensor<f64>) -> tensor<f32>
    %119 = "stablehlo.broadcast_in_dim"(%118) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %120 = "stablehlo.add"(%116, %119) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %121 = "stablehlo.multiply"(%12, %120) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %122 = "stablehlo.subtract"(%121, %114) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %123 = "stablehlo.constant"() <{value = dense<0.67679527440947607> : tensor<f64>}> : () -> tensor<f64>
    %124 = "stablehlo.convert"(%123) : (tensor<f64>) -> tensor<f32>
    %125 = "stablehlo.broadcast_in_dim"(%124) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %126 = "stablehlo.add"(%122, %125) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %127 = "stablehlo.subtract"(%126, %114) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %128 = "stablehlo.constant"() <{value = dense<5.000000e-01> : tensor<f32>}> : () -> tensor<f32>
    %129 = "stablehlo.broadcast_in_dim"(%128) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %130 = "stablehlo.multiply"(%129, %127) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %131 = "stablehlo.divide"(%10, %4) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %132 = "stablehlo.subtract"(%131, %8) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %133 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %134 = "stablehlo.broadcast_in_dim"(%133) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %135 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %136 = "stablehlo.broadcast_in_dim"(%135) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %137 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %138 = "stablehlo.broadcast_in_dim"(%137) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %139 = "stablehlo.multiply"(%132, %134) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %140 = "stablehlo.subtract"(%139, %136) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %141 = "stablehlo.constant"() <{value = dense<3.3962320257083865E-9> : tensor<f64>}> : () -> tensor<f64>
    %142 = "stablehlo.convert"(%141) : (tensor<f64>) -> tensor<f32>
    %143 = "stablehlo.broadcast_in_dim"(%142) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %144 = "stablehlo.add"(%140, %143) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %145 = "stablehlo.multiply"(%132, %144) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %146 = "stablehlo.subtract"(%145, %134) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %147 = "stablehlo.constant"() <{value = dense<2.266668990498178E-8> : tensor<f64>}> : () -> tensor<f64>
    %148 = "stablehlo.convert"(%147) : (tensor<f64>) -> tensor<f32>
    %149 = "stablehlo.broadcast_in_dim"(%148) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %150 = "stablehlo.add"(%146, %149) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %151 = "stablehlo.multiply"(%132, %150) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %152 = "stablehlo.subtract"(%151, %144) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %153 = "stablehlo.constant"() <{value = dense<2.0489185894690638E-7> : tensor<f64>}> : () -> tensor<f64>
    %154 = "stablehlo.convert"(%153) : (tensor<f64>) -> tensor<f32>
    %155 = "stablehlo.broadcast_in_dim"(%154) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %156 = "stablehlo.add"(%152, %155) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %157 = "stablehlo.multiply"(%132, %156) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %158 = "stablehlo.subtract"(%157, %150) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %159 = "stablehlo.constant"() <{value = dense<2.8913705208347567E-6> : tensor<f64>}> : () -> tensor<f64>
    %160 = "stablehlo.convert"(%159) : (tensor<f64>) -> tensor<f32>
    %161 = "stablehlo.broadcast_in_dim"(%160) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %162 = "stablehlo.add"(%158, %161) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %163 = "stablehlo.multiply"(%132, %162) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %164 = "stablehlo.subtract"(%163, %156) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %165 = "stablehlo.constant"() <{value = dense<6.8897583469168245E-5> : tensor<f64>}> : () -> tensor<f64>
    %166 = "stablehlo.convert"(%165) : (tensor<f64>) -> tensor<f32>
    %167 = "stablehlo.broadcast_in_dim"(%166) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %168 = "stablehlo.add"(%164, %167) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %169 = "stablehlo.multiply"(%132, %168) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %170 = "stablehlo.subtract"(%169, %162) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %171 = "stablehlo.constant"() <{value = dense<0.0033691164782556943> : tensor<f64>}> : () -> tensor<f64>
    %172 = "stablehlo.convert"(%171) : (tensor<f64>) -> tensor<f32>
    %173 = "stablehlo.broadcast_in_dim"(%172) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %174 = "stablehlo.add"(%170, %173) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %175 = "stablehlo.multiply"(%132, %174) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %176 = "stablehlo.subtract"(%175, %168) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %177 = "stablehlo.constant"() <{value = dense<0.80449041101410879> : tensor<f64>}> : () -> tensor<f64>
    %178 = "stablehlo.convert"(%177) : (tensor<f64>) -> tensor<f32>
    %179 = "stablehlo.broadcast_in_dim"(%178) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %180 = "stablehlo.add"(%176, %179) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %181 = "stablehlo.subtract"(%180, %168) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %182 = "stablehlo.constant"() <{value = dense<5.000000e-01> : tensor<f32>}> : () -> tensor<f32>
    %183 = "stablehlo.broadcast_in_dim"(%182) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %184 = "stablehlo.multiply"(%183, %181) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %185 = "stablehlo.sqrt"(%4) : (tensor<20x20xf32>) -> tensor<20x20xf32>
    %186 = "stablehlo.divide"(%184, %185) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %187 = "stablehlo.constant"() <{value = dense<8.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %188 = "stablehlo.broadcast_in_dim"(%187) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %189 = "stablehlo.compare"(%4, %188) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LE>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %190 = "stablehlo.select"(%189, %130, %186) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    "stablehlo.custom_call"(%190, %3) <{call_target_name = "check.expect_almost_eq", has_side_effect = true}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> ()
    "func.return"(%190) : (tensor<20x20xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x20xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x4BF5E83CCDFC03404E798CC03CBA14BF7D3E0EC038FC70C017A558C06DC8EFBE197C25C01A05BB3F38FF913FA15B933F88FC3BBF2CAF8BC042088C3FB60ABA40BE8B8D40366A493FDBD6E1BE46E8FD3FAE4FB1BF157D39BE6438D6BF0155883F05E0B0BFB1FEA8C01C65DA3F3F6C43401721ECBE0DEB3D3FCCB1FBBDEFD427BF921AE6BE16AF0A4018E5A2C0A4E30E3F6252F1BFE123FABE1B474EC02B1C044038F2B73F3F04BFBF6B39A6C0107A423F205E95C03D24E7BEFFF40640BD7246C0417501C12A66F6BF9FD3523F99969B4002BCDC3F9607903F2F8BA2405A947CC0810287C05DC748BF8B9910C0056BCD3FEE398C3E381450BFA013CC3EDD033FC0CC83D93F29026E3FC80B8EC0FEA47940241012C06F0EEDBF9DCBABBFAC53123FC87C53C0C22D60C0CB2C45C07F5E8F3E4AFA0EC0C9770840E7F88CC02A03563ED10C30BFD12E43C0DCF06BC0422113C02074C4BF961E29C0F1409EC0324B94401A1EE5BFBDE14C3F1A15D74071AB83407666D23F41DA46BF1D6FC13FC0819DBFF5E6F53F6382A3C0466E04C04366014008F46ABFC22AC5BF8631033F23C1A4C0B3358D40A821E63F44D29EC0138B2AC0D88B2FC015B791404FA55C406FBE544087C49EC0650335C08BC0CDBD550384C0C45C9A3ED7A19FBEFC61AC3FD38FB13F9BCAA8BFBCB283C0183A49409EFD41BFA25E2D3FA7C87DBFF9B8B73FE32797C0E2CAF33F5DC35340EFDD03401013B4C0734314C06C01AABF486085C02DDD333F33BDC03FF13F60BF16C98BBFF38DA4BF5797504078F588BEFDEEE13F926D6CBF5E311CBFD029A040DE04DCBF0107833F00BEB63FA8548DC0B780584086943BBF60978CC0245D893F18CCE43FB741DFBFA0973D400602893EBC91C3BD713C253F0149A8C045571E40F079164008AC2DC0654F4E3E6821B840EC6382401EA972C0B3D00940814EA23F8EB34ABE83C4D7BF2C4C344053466D40F38903409FC7613F12A19BC023A3E73F07D85CC0FD3563BF9A04C140ED548D3FF73F61401C7FF840DE8C5EBF2CE39F3FE1E704401FDFE0409C85174044DEF2BFE8F4083F97B0D9BFAF40923FDA352040273E9D3FFE9935BF5B7A0BBFA14900C02BC82D3F079AB6BFE2A405BF87C318C15BC484408CF382C09ED2F53DB7E7EABFC8967C40289074BF1D6ED03FEF88EC4045CD2740A6B2D33FBC5DA5BF17B7E3405CC7553E556022C045D8B2404E96AEBF44EBF4BEFBC672C0E98073401992F23FB6D3AFBF0AF77CBE00F5254010043EC0157DCBBFD73F6CBFCBA3C4BF390EFB3FAEA262BF486EAEBF9C5327C029655340218A403FFFF57DBFE2EBC5BF650970BF204321C0E5BB70BEBDD4A6C0CC39AC408E1F3DC0E6DA0340369E12BF3CAA32404201403EB091D5C07E2089C0C4F52FC07C079D3F648BEFC0B8F012411E160B40C57DEFBF3D9FAEBFD2316BC03F19B0BF96F7B8BE4A7F0240C0D8704078587440E2EBA7BE13BD8D403242F3BF757D32C0DB8DEEC0E157A9BFA3840CC0DF075040EBB951407C8A73BF763408BF818F1FC06BAECAC0BC3212C1DD0526C0195253C0AFE67E3FF1F25340AE91AA3FEC0CA1C0A9CDC4BF6DA418407A508DBF2528D33B73A919C09B990E3F9473D03F5292484002433B40DF85C0BFEC651340831CA03FDD4F104017B2AA405616A83F599E804029107B3FCD8656407D94B040E7D8C3BF0EF459BFC4270DC0C4CAEF3F1BBE6FC0CB7483C027A37C40489C3F3DBCDEDDC0F517814088F7F53EB14CA2BF789982C0AD3F3840E2E117C04DE692BF73749CC087DD563CCE9D5240B5648F401FDB7F3E32314B3EF69E0DBD20A399BF3A66BCBF71F7E93E93C440BFE1768E40FB2464C0D30298BF55CA3CC034A639407D1260C0B33A6F405D705B40A99EF33E36AA06C05FE6BDC0E1A01240A6C3E0C04D1F883E822F88C0FA54103FCA10AA406DFCF3BEEEBB16C0C3499FC039E45240FEC9A8C0556A684041D7C73EE954B6BF2528AABF7BE954C0BEA2F03E354D653F6EE586C0A621C83F5840B4BFFED5BCBF511937BF067DE73EA42990C07C0B2EC05796283F4D62C63FCFA4B1407006A8BF288A94401A7520401021B4BF599D7CBE260A0DBF9834DABF5F5D57BF2FD96E3F97F260405ED32E3F1A5F7540A9553BC041B3B4C0B15B33BFF79EA0C00309D3BFF82FAFBFD317C73FE2825E40903120408AFA99BF420939C0FD9263C0E153D03F0C0FEF3F97B17A3F07650841B0217D40D49CD6BF894B86BD7AAF8CBD"> : tensor<20x20xf32>}> : () -> tensor<20x20xf32>
    "func.return"(%1) : (tensor<20x20xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x20xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x74DF783FEF0E9B3E0A89493E23891B3F473F943E001C5B3EB86F683E802D293F5A8F873E3D24BF3EAC5CDD3EA22DDC3E61F60B3F58264A3E52B9E23E3E832D3EC6B5483EB24E073F54CD2C3FFFBD9E3E8461C53E4957573F8003B03EDD32E63E50ACC53EAE8E363E14F5AD3E9957763E471D2A3FFF450B3F1840633F0A91133FDEAD2B3F6983963EC0283A3E0C221E3F80B5A33E6492263FCBEB6E3EBEF89A3E5A10C13E46B6BC3EB72A383E73AC093F4AFC423E68682B3FC9FE983E0A34743ED403123E74A7A13E8139043F90C53E3E1BD5AC3E4B19DF3E895F3A3E9A88553EB2E84D3E6085073FFFCC923EEA90B43E0E56463F221C053FE8BC323F3290793EB662AE3E7B26F83E1C54483E63E7563E33EC913EE47DA53E7A27C93E16981C3F14986B3E6B0C643EB418753ED849453FD6CA933E9EF9973E7C26493E0CFE513F2C5F103FA683763EC6AD5D3EBE4A913E4887B93E52DA853E920C3D3EFABD433E87F4A83EBA27063FE4CC203EDABA503EC6F3B13EFC2B083F9647BB3E47C9D33E23DAA13E5DC9393ED1BE9A3E06EC9C3EB2E0F93EE91EB93EF094233FD509393EBCF7483E7E80A83E34B03C3E9233853E69FC823E8C98453E0713663E0ED06A3EEAB83C3E10B0803E861D683F396F503EECB1413FC602403F7CBEC83EAD36C53E9048CB3E91B4503EB949723EADD7093F2A66113F249EEF3EA234C13ECBBE413E3AB3A23E026C6B3ECF249B3EC585303E68A1903E986ACA3E27464F3E14EF0E3F49B0BB3E0411003FB7F3E23E4D61CE3E0E6D6D3E646F473F7466AA3E6E0AF93EDE53183FF7D73B3EA42CAD3EDE63EB3E79D4C13EF8DF483E8D85683EB11B0C3FC271493ECA37E53E5E19A93E17A4AB3E7AA17A3E276B473F2E37693F939A143F2FF7363E211D8B3E385D8F3E8ACD833E5C6A533FB3752E3E9AD7513EA8465A3EDF14973EA511D03E6716543F7A3EAF3E50FB803E8AFD5C3E7E609B3E0E38FF3EB4BE3E3EB4D5A73E91F5653EBE5EFE3EF62E2A3EF487E13EF071633E2C28153EFC93003F5DECD13E77699A3EF01B1D3EE7C68E3E5713A33EDADA203FDC4CAE3E7223DD3E38288A3EC7FED33EB8490E3F0EB01F3FFDBD9D3E7B3D113F7CEBC13E8169223F110E063E45CA4F3E2A5A513EFBDC633F4469A63E7B87553E1987F43E5AF9B23E1A07193ED776863E9949B13E38C7CD3E79151C3E0909523FA212893EA229313E0A39C73E62E0273FE4375A3E3FDC593E6732A33E1C61C63EFA064B3FA055873EBD4F7A3EF69AB53E4C24F93EFC6BB93EB8D49F3EEDB5FE3E6D54C73ED2AF863EDAA66B3E59590A3F8786EF3E2DB1B83E6C04F73E9AA0893E622C4D3F8ACF373EB8BB343E58FC7A3EF6269B3E0B771C3FEEA8813E9719563F0865213E1B2E4C3EC5CE823E252AD43E8206183E06C4083E8C40963EB577A43EF032C73E1A115E3E1932C63E503F383FB11F9C3EC22D5B3E9472593E31693D3F1590483EADEAA23EB2BB813E635A183E66E3CA3E5E55953E1AC96D3EF0B36C3ED415F53EB434213FC87C8A3E10E0253E861F093E984D873EC9B26B3E7E09EF3E5E4E6B3E5604CA3E9E4A3B3E0A54B93ED3278E3E048CE13EB85B7E3FCE988D3E82431E3F7FF6B23E4EBC723EE2687C3EF6D0BB3E7822913EFEBFD13EAEF9923E1897353E5ACACB3EBE69533E610BF13EB3B7693E365C323E72E0B93EB0FB013F03EE943EA157A43EDDBB5B3E0CEA503EC481553E106F743FB2361E3E05FD523EFA9C273F0513D03EBCA8513E7EC47E3E80938E3E4593DC3ED8343E3EEBAC7C3FF3236C3ECE4F473E7E864A3FF4FE533FB65F773FE2E5D63E524BBE3E50AC2A3FE3440A3FDA02483EAAD6613ECA3CD83E123D7B3ECEA97D3EDB1B643E4EFE5B3E70C7663E3934283FDB31993EBDA52B3E6296913EF2251D3ED7B7473F7CF14C3ED87B1D3F5AF2353E8E1C283FFD378F3EA8643C3EADF76B3EEBAC363ECE875F3E3EEF333FC817C23E194FCA3E6FB56A3EDFF5283FF823FD3EC1004E3E6473B73E6170C33E2B07BE3E62BC0D3F3D512B3FFABC463EA0A3833E6A44133F356EB83E6FCB313ED6D5CB3E6B91433E3A088A3ECE84C33E96164B3FF4F81E3FA00CAE3ECFC9023F04AEF73E689D633EDCD6103FA8F2583E805A7C3E1033303E7D1F0F3FE48E3B3E4CA0B13E38D0C63E0408B83E70FF643E642A8A3E819ED63E18257E3EFE26623E1107B33EF4A5A43E553DF13EF91F0E3E6147553E54D1AF3E6E02703F6B496F3F"> : tensor<20x20xf32>}> : () -> tensor<20x20xf32>
    "func.return"(%0) : (tensor<20x20xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

